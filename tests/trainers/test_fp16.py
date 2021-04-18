# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import torch
from tests.test_utils import SimpleModel, skip_if_no_cuda


class SimpleModelWithFp16Assert(SimpleModel):
    def forward(self, sample_list):
        batch_tensor = sample_list[list(sample_list.keys())[0]]
        # Start should be fp32
        assert batch_tensor.dtype == torch.float32
        batch_tensor = self.linear(batch_tensor)

        # In between operation should be fp16
        assert batch_tensor.dtype == torch.float16
        loss = torch.sum(batch_tensor)

        # Sum should convert it back to fp32
        assert loss.dtype == torch.float32

        model_output = {"losses": {"loss": loss}}
        return model_output
