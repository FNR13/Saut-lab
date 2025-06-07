import matplotlib.pyplot as plt
import numpy as np

real_landmarks_map2 = np.array([
    (0, -0.30),      # 297
    (-0.88, 0.26),   # 557
    (-0.70, 1.23),   # 934
    (0.80, 1.15),    # 
    (0.80, 2.85),    # 
    (0.80, 4.21),    # 
    (0.80, 6.06),    # 
    (0.50, 8.01),    # 433
    (-0.73, 8.01),   # 63
    (-0.90, 5.93),   # 
    (-0.62, 5.43),   # 
    (-0.41, 4.45),   # 
    (-0.41, 2.33),   # 
    (-2.00, 6.26),   # 
    (-2.87, 7.99),   # 844
])

fig, ax = plt.subplots()
ax.scatter(real_landmarks_map2[:, 0], real_landmarks_map2[:, 1], c='blue', marker='o', label='Landmarks')

# Annotate each landmark with its coordinates
for (x, y) in real_landmarks_map2:
    ax.annotate(f'({x:.2f}, {y:.2f})', (x, y), textcoords="offset points", xytext=(5,5), ha='left', fontsize=8)

ax.set_title('MAP2 Landmarks')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.grid(True)
ax.legend()
plt.axis('equal')
plt.show()