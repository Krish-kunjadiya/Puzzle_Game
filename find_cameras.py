import cv2

index = 0
while True:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.read()[0]:
        break
    print(f"Camera Index {index}: Works")
    cap.release()
    index += 1

print(f"\nFound {index} cameras.")
