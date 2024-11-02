import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y1 = [10, 20, 25, 30, 35]  # Data for the first y-axis
y2 = [1, 4, 9, 16, 25]     # Data for the second y-axis

# Create figure and plot on the first y-axis
fig, ax1 = plt.subplots()
ax1.plot(x, y1, 'g-', label='Y1 Data')  # 'g-' sets the line color to green
ax1.set_xlabel('X Axis')
ax1.set_ylabel('Y1 Axis', color='g')
ax1.tick_params(axis='y', labelcolor='g')

# Create the second y-axis sharing the same x-axis
ax2 = ax1.twinx()
ax2.plot(x, y2, 'b-', label='Y2 Data')  # 'b-' sets the line color to blue
ax2.set_ylabel('Y2 Axis', color='b')
ax2.tick_params(axis='y', labelcolor='b')

# Optional: add legends for clarity
fig.tight_layout()  # Ensures the layout fits well
plt.show()