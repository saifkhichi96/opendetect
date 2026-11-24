from types import SimpleNamespace

import numpy as np

from opendetect.detector import Detector
from opendetect.models.rfdetr import RFDETRModel


def test_dynamic_rfdetr_batches_are_bounded_and_keep_image_sizes():
    backend = RFDETRModel.__new__(RFDETRModel)
    backend.session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(shape=["batch", 3, 32, 32])]
    )
    backend.preprocess_rgb_frame = lambda image: (
        np.zeros((1, 3, 32, 32)),
        np.array([image.shape[:2]]),
    )
    calls = []

    def forward(tensor):
        calls.append(len(tensor))
        return ()

    backend.forward = forward
    backend.postprocess = lambda outputs, sizes: [
        {"size": tuple(size)} for size in sizes
    ]
    detector = Detector.__new__(Detector)
    detector._backend = backend
    result = detector.predict_batch(
        [np.zeros((height, 40, 3), np.uint8) for height in range(10, 15)], batch_size=2
    )
    assert calls == [2, 2, 1]
    assert [item["size"] for item in result] == [
        (height, 40) for height in range(10, 15)
    ]
