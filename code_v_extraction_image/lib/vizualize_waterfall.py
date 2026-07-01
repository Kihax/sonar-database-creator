import matplotlib.pyplot as plt
def vizualize_waterfall(image):
    plt.imshow(image, cmap='gray', vmin=0, vmax=5, aspect='auto')
    plt.grid(True)
    plt.title('HF Sonar Waterfall')
    plt.show()