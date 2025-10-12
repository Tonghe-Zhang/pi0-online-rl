



This is what is contained in the `next_obs` returned by `PutOnPlateInScene25Single-v1` environment. 

```python
next_obs=
{
  "agent": {
    "qpos": {
      "shape": [2, 8],
      "type": "torch.float32"
    },
    "qvel": {
      "shape": [2, 8],
      "type": "torch.float32"
    },
    "controller": {
      "arm": {
        "target_pose": {
          "shape": [2, 7],
          "type": "torch.float32"
        }
      }
    }
  },
  "extra": {
    "shape": null,
    "type": "dict"
  },
  "sensor_param": {
    "3rd_view_camera": {
      "extrinsic_cv": {
        "shape": [2, 3, 4],
        "type": "torch.float32"
      },
      "cam2world_gl": {
        "shape": [2, 4, 4],
        "type": "torch.float32"
      },
      "intrinsic_cv": {
        "shape": [2, 3, 3],
        "type": "torch.float32"
      }
    }
  },
  "sensor_data": {
    "3rd_view_camera": {
      "segmentation": {
        "shape": [2, 480, 640, 1],
        "type": "torch.int32"
      },
      "rgb": {
        "shape": [2, 480, 640, 3],
        "type": "torch.uint8"
      }
    }
  }
}
```