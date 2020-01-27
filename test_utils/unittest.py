
import numpy as np
import matplotlib.pyplot as plt
import os

from pixor_utils.post_processing import nms_bev

def test_pc_encoder(pc_encoder, pts):
    assert pc_encoder is not None, 'Test PC Encoder -> pc_encoder is None'
    assert pts is not None, 'Test PC Encoder -> Point Cloud is None'
    assert pts.shape[1] == 3, 'Test PC Encoder -> Point Cloud shape[1] expected value: 3, got {0}'.format(pts.shape[1])
    reflectance = None
    encoded_pc = pc_encoder.encode(pts, reflectance)
    plt.imshow(np.sum(encoded_pc, axis=2) > 0)
    plt.savefig(os.path.dirname(__file__) + '/test_encoded_pc.png')
    assert encoded_pc.shape == (800, 700, 35), 'Test PC Encoder -> Something wrong happened while Encoding PC \
                                                expecting shape (800, 700, 35), got {0}'.format(encoded_pc.shape)
    print('PC encoder test status: OK')
    print('---------------------------------------------------------------')
    return True
    
def test_target_encoder(target_encoder, boxes):
    assert boxes is not None, 'Test Target Encoder -> boxes is None'
    assert target_encoder is not None, 'Test Target Encoder -> target_encoder is None'
    target = np.squeeze(target_encoder.encode(boxes))
    plt.imshow(target[...,0])
    plt.savefig(os.path.dirname(__file__) + '/test_encoded_target.png')
    assert target.shape == (200, 175, 7), 'Test Target Encoder -> Something wrong happened while Encoding \
                                            boxes, expecting shape (200, 175, 7), got {0}'.format(target.shape)
    decoded_boxes = target_encoder.decode(target, 0.5)
    filtered_boxes = nms_bev(decoded_boxes, 0.1, 10000, 0, False)
    assert len(filtered_boxes) == len(boxes), 'Test Target Encoder -> length of GT boxes and Pred boxes do not match'
    matched_boxes = []
    for p_box in filtered_boxes:
        cur_c_x, cur_c_y = p_box.x, p_box.z
        eps = 0.1
        for gt_box in boxes:
            diff_c_x, diff_c_y = np.abs(cur_c_x - gt_box.x), np.abs(cur_c_y - gt_box.z)
            if diff_c_x < eps and diff_c_y < eps:
                matched_boxes.append(gt_box)
                break
    assert len(filtered_boxes) == len(matched_boxes), 'Test Target Encoder -> problem matching filtered bounding boxes with GT'
    print('Target Encoder test status: OK')
    print('---------------------------------------------------------------')
    return True
