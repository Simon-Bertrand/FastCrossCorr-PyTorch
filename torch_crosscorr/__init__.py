from typing import Literal

import torch
import torch.nn.functional as F


class FastNormalizedCrossCorrelation(torch.nn.Module):
    EPS_NORM_ENERGY_SQRT = 1e-8
    EPS_NORM_NULL_THRESHOLD_DENOM = 1e-8
    """
    This class represents a module for performing fast normalized
    cross-correlation.
    The input image must be a real tensor of shape (B, C, H, W) and the
    template must be a real tensor of shape (B, C, h, w).

    Args:
        statistic (Literal["corr", "ncorr"]): The type of correlation
            statistic to compute. Must be either "corr" for correlation or
            "ncorr" for normalized correlation.
        method (Literal["fft", "spatial", "naive"]): The method to use for
            computing the cross-correlation. Must be one of "fft" for
            FFT-based method, "spatial" for spatial method, or "naive" for
            naive method.

    Raises:
        ValueError: If the statistic is not "corr" or "ncorr".
        ValueError: If the method is not "fft", "spatial", or "naive".

    Methods:
        forward(im, template): Performs the forward pass of the module.

    Examples:
        # Create an instance of FastNormalizedCrossCorrelation module
        fnc = FastNormalizedCrossCorrelation(statistic="corr", method="fft")

        # Perform forward pass
        output = fnc(input_image, template)
    """

    def __init__(
        self,
        statistic: Literal["corr", "ncorr"],
        method: Literal["fft", "spatial", "naive"],
        dtype=None,
    ):
        super().__init__()
        self.crossCorrelation = self._chooseMethod(method)
        match statistic:
            case "corr":
                # If the padding fill value is zero, the correlation near
                # borders is lower as a lot of zeros are present
                # We try to avoid that by using padding mode reflect to weight
                # more borders values
                self.normalize = False
            case "ncorr":
                self.normalize = True
            case _:
                raise ValueError(
                    f"Statistic must be 'corr' or 'ncorr',\
got {statistic}"
                )

        self.output_dtype = dtype

    def _computeRectangleSum(self, intIm, ii, jj, padWl, padWr, padHt, padHb):
        """
        Computes the sum of values in a rectangular region of the input tensor.
        Returns:
            torch.Tensor: The sum of values in the rectangular region.
        """
        return (
            intIm[:, :, ii - padHt - 1, jj - padWl - 1]
            + intIm[:, :, ii + padHb, jj + padWr]
            - intIm[:, :, ii - padHt - 1, jj + padWr]
            - intIm[:, :, ii + padHb, jj - padWl - 1]
        )

    def _chooseMethod(self, method):
        """
        Chooses the appropriate method for computing the cross-correlation
        based on the given method argument.

        Returns:
            function: The chosen method for computing the cross-correlation.
        """
        match method:
            case "fft":
                return self._crossCorrFFT
            case "spatial":
                return self._crossCorrSpatial
            case "naive":
                return self._crossCorrNaive
            case _:
                raise ValueError(
                    f"Method must be 'fft' or 'spatial', 'naive',\
        got {method}"
                )

    # FFT METHOD
    def _crossCorrFFT(self, imCentered, template, *_):
        # We flip the template because we want to cross correlate
        # (and not convolve) the image with the template.
        # Note that, we don't need the image mean value to be zero if the
        # template is zero mean, check page 2 right side :
        # Lewis, J.P.. (1994). Fast Template Matching. Vis. Interface. 95.
        # http://scribblethink.org/Work/nvisionInterface/vi95_lewis.pdf
        return torch.fft.irfft2(
            torch.fft.rfft2(
                imCentered,
                s=(
                    padded_shape := (
                        (
                            imCentered.size(-2)
                            + template.size(-2)
                            - template.size(-2) % 2
                        ),
                        (
                            imCentered.size(-1)
                            + template.size(-1)
                            - template.size(-1) % 2
                        ),
                    )
                ),
            )
            * torch.fft.rfft2(torch.flip(template, dims=(-2, -1)), padded_shape)
        )[
            ...,
            (hH := template.size(-2) // 2) : -(hH),
            (hW := template.size(-1) // 2) : -(hW),
        ]

    # NAIVE METHOD
    def _crossCorrNaive(self, imCentered, template, *padding):
        return (
            torch.nn.functional.pad(
                imCentered,
                padding,
            )
            .unfold(-2, template.size(-2), 1)
            .unfold(-2, template.size(-1), 1)
            .flatten(-2, -1)
            * template.flatten(-2, -1).unsqueeze(2).unsqueeze(2)
        ).sum(dim=-1)

    # SPATIAL METHOD
    def _crossCorrSpatial(self, imCentered, template, *_):
        # Conv2D is a cross-correlation in PyTorch, not real convolutions
        return (
            F.conv2d(
                imCentered.flatten(0, 1).unsqueeze(0),
                template.flatten(0, 1).unsqueeze(1),
                padding="same",
                groups=imCentered.size(1) * imCentered.size(0),
            )
            .unflatten(1, imCentered.shape[:2])
            .squeeze(0)
        )

    def forward(self, im, template):
        """
        Performs the forward pass of the module.
        Note : When using ncorr and integral images to compute the normalization,
        one should consider the accumulated errors from the cumsum applied twice on float32.
        To summarize, it is better to use f64 when using ncorr in order to avoid underflow/overflow.

        Args:
            im (torch.Tensor): The input image tensor of shape (B, C, H, W).
            template (torch.Tensor): The template tensor of shape (B, C, h, w).

        Returns:
            torch.Tensor: The output tensor of shape (B, C, H, W).
        """
        # Image must be a real tensor of shape (B, C, H, W) and template must
        # be a real tensor of shape (B, C, h, w)
        # Output is a real tensor of shape (B, C, H, W)
        padding = [
            (padWl := (template.size(-1) - 1) // 2),
            (padWr := (padWl + 1 - template.size(-1) % 2)),
            (padHt := ((template.size(-2) - 1) // 2)),
            (padHb := (padHt + 1 - template.size(-2) % 2)),
        ]

        cache = torch.nn.functional.pad(
            im.to(dtype=self.output_dtype),
            [padWl + 1, padWr, padHt + 1, padHb],
            mode="reflect",
        )
        cache[:, :, 0, :] = 0.0
        cache[:, :, :, 0] = 0.0
        iiSlice = slice(1 + padHt, 1 + cache.size(-2) - padHb - 1)
        jjSlice = slice(1 + padWl, 1 + cache.size(-1) - padWr - 1)
        ii, jj = torch.meshgrid(
            torch.arange(iiSlice.start, iiSlice.stop, device=im.device),
            torch.arange(jjSlice.start, jjSlice.stop, device=im.device),
            indexing="ij",
        )

        imCentered = im - self._computeRectangleSum(
            cache.cumsum(-1).cumsum(-2), ii, jj, *padding
        ) / (template.size(-2) * template.size(-1))

        templateCentered = template - template.mean(dim=(-2, -1), keepdim=True)
        numerator = self.crossCorrelation(imCentered, templateCentered, *padding)

        if self.normalize:
            cache[:, :, 1:, 1:] = 0.0
            cache[:, :, iiSlice, jjSlice] = imCentered.to(self.output_dtype).pow(2)
            energy = self._computeRectangleSum(
                cache.cumsum(-1).cumsum(-2), ii, jj, *padding
            )
            energySqr = (
                energy.clamp(min=0).where(energy > self.EPS_NORM_ENERGY_SQRT, 0)
            ).sqrt()
            templateNorm = templateCentered.norm(p=2, dim=(-2, -1), keepdim=True)
            denom: torch.Tensor = energySqr * templateNorm
            return torch.where(
                denom >= self.EPS_NORM_NULL_THRESHOLD_DENOM & ~denom.isnan(),
                numerator / denom,
                0,
            )

        return numerator
