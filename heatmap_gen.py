import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def draw_grid(image_path, output_path, rows=8, cols=8, line_color=(128, 128, 128), line_width=2):
    # Open the image
    image = Image.open(image_path)
    width, height = image.size

    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Calculate the width and height of each cell
    cell_width = width / cols
    cell_height = height / rows

    # Draw horizontal lines
    for i in range(1, rows):
        y = i * cell_height
        draw.line([(0, y), (width, y)], fill=line_color, width=line_width)

    # Draw vertical lines
    for i in range(1, cols):
        x = i * cell_width
        draw.line([(x, 0), (x, height)], fill=line_color, width=line_width)

    # Save the result
    image.save(output_path)


# Example usage
draw_grid('/home/amos/PycharmProjects/COD_TEST/visualzation/reconstructed_image_b0.jpg', 'output_image_with_grid.jpg')

