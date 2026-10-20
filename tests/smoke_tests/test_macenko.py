import cv2
from src.normalization import stainNorm_Macenko

print("1. Imports successful!", flush=True)

target = cv2.cvtColor(
    cv2.imread("/content/target.png"), cv2.COLOR_BGR2RGB
)
source = cv2.cvtColor(
    cv2.imread("/content/source.png"), cv2.COLOR_BGR2RGB
)

print("2. Images loaded!", flush=True)

macenko = stainNorm_Macenko.normalizer()
macenko.fit(target)

print("3. Target fitted!", flush=True)

result = macenko.transform(source)

print("4. Source transformed!", flush=True)

cv2.imwrite(
    "/content/macenko_output.png",
    cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
)

print("Output shape:", result.shape)
print("Output dtype:", result.dtype)
print("MACENKO TEST SUCCESSFUL!", flush=True)
