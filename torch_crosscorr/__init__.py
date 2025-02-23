from typing import Literal

import torch
import torch.nn.functional as F


class FastNormalizedCrossCorrelation(torch.nn.Module):
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
        padding: Literal["same", "valid"] = "same",
        center=True,
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

        if padding not in ["same", "valid"]:
            raise ValueError(f"padding={padding} not supported")
        self.padding = padding
        self.output_dtype = dtype
        self.center = True

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
        hH = template.size(-2) // 2
        hW = template.size(-1) // 2
        match self.padding:
            case "same":
                validSupportI = slice(hH, -(hH) + imCentered.size(-2) % 2)
                validSupportJ = slice(hW, -(hW) + imCentered.size(-1) % 2)
            case "valid":
                validSupportI = slice(
                    template.size(-2) - 1,
                    -(template.size(-2))
                    + template.size(-2) % 2
                    + imCentered.size(-2) % 2,
                )
                validSupportJ = slice(
                    template.size(-1) - 1,
                    -(template.size(-1))
                    + template.size(-1) % 2
                    + imCentered.size(-1) % 2,
                )
            case _:
                raise ValueError(f"padding={self.padding} not supported")

        return torch.fft.irfft2(
            torch.fft.rfft2(
                imCentered,
                s=(
                    padded_shape := (
                        (
                            imCentered.size(-2)
                            + template.size(-2)
                            - template.size(-2) % 2
                            - imCentered.size(-2) % 2
                        ),
                        (
                            imCentered.size(-1)
                            + template.size(-1)
                            - template.size(-1) % 2
                            - imCentered.size(-1) % 2
                        ),
                    )
                ),
            )
            * torch.fft.rfft2(torch.flip(template, dims=(-2, -1)), padded_shape)
        )[..., validSupportI, validSupportJ]

    # NAIVE METHOD
    def _crossCorrNaive(self, imCentered, template, *padding):
        return (
            (
                torch.nn.functional.pad(
                    imCentered,
                    padding,
                )
                if self.padding == "same"
                else imCentered
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
                padding=self.padding,
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
            (
                padWr := (padWl + 1 - template.size(-1) % 2)
            ),  # The trick here "+ 1 - template.size(-1) % 2" is to add 1 when the template size is even
            (padHt := ((template.size(-2) - 1) // 2)),
            (padHb := (padHt + 1 - template.size(-2) % 2)),
        ]
        im = im.to(dtype=self.output_dtype)
        # Cache used to compute the integral image, we need adding 1 row and 1 column of zeros at the top left
        # The cache is used a second time for "ncorr" to compute the energy of the image
        # It avoids double allocation
        cache = torch.nn.functional.pad(
            im,
            [padWl + 1, padWr, padHt + 1, padHb],
            mode="constant",
        )
        cache[:, :, 0, :] = 0.0
        cache[:, :, :, 0] = 0.0
        iiSlice = slice(1 + padHt, cache.size(-2) - padHb)
        jjSlice = slice(1 + padWl, cache.size(-1) - padWr)
        ii, jj = torch.meshgrid(
            torch.arange(iiSlice.start, iiSlice.stop, device=im.device),
            torch.arange(jjSlice.start, jjSlice.stop, device=im.device),
            indexing="ij",
        )  # Get the cache valid support indices
        if self.center:
            imCentered = im - self._computeRectangleSum(
                cache.cumsum(-1).cumsum(-2), ii, jj, *padding
            ) / (
                template.size(-2) * template.size(-1)
            )  # Center the image using integral image
            templateCentered = template - template.mean(
                dim=(-2, -1), keepdim=True
            )  # Center the template using its mean
        else:
            imCentered = im
            templateCentered = template

        if not self.normalize:
            return self.crossCorrelation(
                imCentered, templateCentered, *padding
            )  # Compute the cross-correlation
        else:
            cache[:, :, 1:, 1:] = 0.0  # Reset cache values
            cache[:, :, iiSlice, jjSlice] = imCentered.to(dtype=self.output_dtype).pow(
                2
            )  # Insert image.pow(2) in cache

            energySqr = (
                self._computeRectangleSum(cache.cumsum(-1).cumsum(-2), ii, jj, *padding)
                .clamp(min=0)
                .sqrt()
            )  # Compute energy using integral image of image.pow(2)
            denom: torch.Tensor = energySqr * templateCentered.norm(
                p=2, dim=(-2, -1), keepdim=True
            )  # Compute the denominator with the template L2 norm
            if self.padding == "valid":
                denom = denom[
                    ...,
                    padHt : padHt + imCentered.size(-2) - template.size(-2) + 1,
                    padWl : padWl + imCentered.size(-1) - template.size(-1) + 1,
                ]  # Crop the denominator if padding is valid

            numerator = self.crossCorrelation(
                imCentered / energySqr,
                templateCentered
                / templateCentered.norm(p=2, dim=(-2, -1), keepdim=True),
                *padding,
            )
            return numerator
        return numerator
