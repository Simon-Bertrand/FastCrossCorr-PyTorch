from functools import lru_cache
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
        findArgmax(x): Finds the indices of the maximum values along the last
            dimension of the input tensor.
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

    @staticmethod
    def findArgmax(x):
        """
        Finds the indices of the maximum values along the last dimension of
        the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The indices of the maximum values.
        """
        aMax = x.flatten(-2, -1).argmax(dim=-1)
        return torch.stack([aMax // x.size(-1), aMax % x.size(-1)])

    @lru_cache(maxsize=2)
    def _nextFastLen(self, size):
        """
        Computes the next fast length for FFT-based method.

        Args:
            size (int): The current size.

        Returns:
            int: The next fast length.
        """
        next_size = size
        while True:
            remaining = next_size
            for n in (2, 3, 5):
                while (euclDiv := divmod(remaining, n))[1] == 0:
                    remaining = euclDiv[0]
            if remaining == 1:
                return next_size
            next_size += 1

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
                return self.crossCorrFFT
            case "spatial":
                return self.crossCorrSpatial
            case "naive":
                return self.crossCorrNaive
            case _:
                raise ValueError(
                    f"Method must be 'fft' or 'spatial', 'naive',\
        got {method}"
                )

    # FFT METHOD
    def crossCorrFFT(self, imCentered, template, padWl, padWr, padHt, padHb):
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
                        self._nextFastLen(imCentered.size(-2) + template.size(-2) - 1),
                        self._nextFastLen(imCentered.size(-1) + template.size(-1) - 1),
                    )
                ),
            )
            * torch.fft.rfft2(torch.flip(template, dims=(-1, -2)), padded_shape)
        )[
            ...,
            padHt : padHt + imCentered.size(-2),
            padWl : padWl + imCentered.size(-1),
        ]

    # NAIVE METHOD
    def crossCorrNaive(self, imCentered, template, *padding):
        return (
            torch.nn.functional.pad(
                imCentered,
                padding,
            )
            .unfold(2, template.size(-2), 1)
            .unfold(3, template.size(-1), 1)
            .flatten(-2, -1)
            * template.flatten(-2, -1).unsqueeze(2).unsqueeze(2)
        ).sum(dim=-1)

    # SPATIAL METHOD
    def crossCorrSpatial(self, imCentered, template, *_):
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
            im, [padWl + 1, padWr, padHt + 1, padHb], mode="reflect"
        )
        cache[:, :, 0, :] = 0
        cache[:, :, :, 0] = 0

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
            cache[:, :, 1:, 1:] = 0
            cache[:, :, iiSlice, jjSlice] = imCentered.pow(2)
            energy = self._computeRectangleSum(
                cache.cumsum(-1).cumsum(-2), ii, jj, *padding
            )

            denom = energy.sqrt() * (
                templateCentered.norm(p=2, dim=(-2, -1), keepdim=True)
            )

            return torch.where(denom.abs() < 1e-6, 0, numerator / denom)

        return numerator
