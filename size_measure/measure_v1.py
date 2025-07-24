import cv2
import numpy as np

def detect_object_mask_refine(image_path, max_iter=10, tol=0.02):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图片: {image_path}")
        return

    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (10, 10, w - 20, h - 20)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # 第一次 grabcut 全图初始化
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    prev_area = None
    final_rect = None

    for i in range(max_iter):
        contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"⚠️ 第{i}轮无轮廓，终止")
            break

        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest)
        min_rect = cv2.minAreaRect(hull)
        final_rect = min_rect

        w_rect, h_rect = min_rect[1]
        area = w_rect * h_rect

        if prev_area:
            ratio = area / prev_area
            if abs(1 - ratio) < tol:
                print(f"✅ 收敛于第{i}轮，面积稳定")
                break
        prev_area = area

        # 用新的 mask 区域 refine grabcut
        # 方式：绘制最小矩形包围盒作为 probable foreground
        mask[:] = 0
        box = cv2.boxPoints(min_rect).astype(int)
        cv2.drawContours(mask, [box], -1, 3, -1)  # 3 = probable fg
        cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # 最终输出
    if final_rect is not None:
        w_final, h_final = final_rect[1]
        area_final = w_final * h_final
        print(f"📏 最终尺寸：宽 = {w_final:.2f}px，高 = {h_final:.2f}px，面积 = {area_final:.2f} 像素²")

        box = cv2.boxPoints(final_rect).astype(int)
        output = img.copy()
        cv2.drawContours(output, [box], 0, (0, 255, 0), 2)
        cv2.imwrite("detected_image.jpg", output)
        print("💾 已保存到 detected_image.jpg")
    else:
        print("❌ 未获得有效轮廓")

# 运行
detect_object_mask_refine("small_01.jpg")
