def generate_shapes(num_imgs, destination, show_animation):
    """
        Generate a defined number of images that contains eight geometric
        shapes (Square, Triangle, Circle, Star, Polygon, Pentagon, Heptagon,
        Hexagon).

        Each shape is drawn randomly on a 200x200 image. Each image is drawn
        with the following parameters which their value is selected randomly
        and independently :

            - Image's background color
            - Shape's filling color
            - Shape's rotation angle (between -180° and 180°)
            - Center of the circumscribed circle of a shape

    """
    from generator import GeometricShapes
    generator = GeometricShapes(
        num_imgs=num_imgs, destination=destination, animation=show_animation
    )
    generator.generate_all_splits()


if __name__ == '__main__':
    DEST_DIR = "Data"
    num_imgs = {'train': 1000,
                'valid': 100,
                'test': 100}
    generate_shapes(num_imgs, DEST_DIR, False)
