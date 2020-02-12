import cv2
from utils.common import mask2polygon
import numpy as np
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import colorsys
from .color_map import random_color

_SMALL_OBJECT_AREA_THRESH = 1000


def instance_visualizer(img, boxes=None, labels=None, assigned_colors=None, scale=1):
    # boxes: numpy array; shape: [-1, 4]
    # labels: list(str)
    num_instances = None
    if boxes is not None:
        num_instances = len(boxes)

    vis_img = VisImage(img, scale=scale)
    default_font_size = max(
        np.sqrt(vis_img.height * vis_img.width) // 90, 10 // scale
    )

    if labels is not None:
        assert len(labels) == num_instances
    if assigned_colors is None:
        assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(num_instances)]
    if num_instances == 0:
        return vis_img

    areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)

    sorted_idxs = np.argsort(-areas).tolist()
    # Re-order overlapped instances in descending order.
    boxes = boxes[sorted_idxs] if boxes is not None else None
    labels = [labels[k] for k in sorted_idxs] if labels is not None else None
    assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]
    for i in range(num_instances):
        color = assigned_colors[i]
        draw_box(vis_img, boxes[i], default_font_size, edge_color=color)
        if labels is not None:
            # first get a box
            x0, y0, x1, y1 = boxes[i]
            text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
            horiz_align = "left"
            instance_area = (y1 - y0) * (x1 - x0)
            if (
                instance_area < _SMALL_OBJECT_AREA_THRESH * vis_img.scale
                or y1 - y0 < 40 * vis_img.scale
            ):
                if y1 >= vis_img.height - 5:
                    text_pos = (x1, y0)
                else:
                    text_pos = (x0, y1)

            height_ratio = (y1 - y0) / np.sqrt(vis_img.height * vis_img.width)
            lighter_color = _change_color_brightness(color, brightness_factor=0.7)
            font_size = (
                    np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                    * 0.5
                    * default_font_size
            )

            draw_text(
                vis_img,
                labels[i],
                text_pos,
                font_size = font_size,
                color=lighter_color,
                horizontal_alignment=horiz_align,
            )
    return vis_img


def draw_text(visimage, text, position, font_size=None, color="g", horizontal_alignment="center", rotation=0):
    color = np.maximum(list(mplc.to_rgb(color)), 0.2)
    color[np.argmax(color)] = max(0.8, np.max(color))
    x, y = position
    visimage.ax.text(
        x,
        y,
        text,
        size=font_size * visimage.scale,
        family="sans-serif",
        bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
        verticalalignment="top",
        horizontalalignment=horizontal_alignment,
        color=color,
        zorder=10,
        rotation=rotation,
    )
    return visimage


def draw_box(visimage, box_coord, default_font_size, alpha=0.5, edge_color="g", line_style="-"):
    """
    Args:
        box_coord (tuple): a tuple containing x0, y0, x1, y1 coordinates, where x0 and y0
            are the coordinates of the image's top left corner. x1 and y1 are the
            coordinates of the image's bottom right corner.
        alpha (float): blending efficient. Smaller values lead to more transparent masks.
        edge_color: color of the outline of the box. Refer to `matplotlib.colors`
            for full list of formats that are accepted.
        line_style (string): the string to use to create the outline of the boxes.

    Returns:
        output (VisImage): image object with box drawn.
    """

    x0, y0, x1, y1 = box_coord
    width = x1 - x0
    height = y1 - y0

    linewidth = max(default_font_size / 4, 1)

    visimage.ax.add_patch(
        mpl.patches.Rectangle(
            (x0, y0),
            width,
            height,
            fill=False,
            edgecolor=edge_color,
            linewidth=linewidth * visimage.scale,
            alpha=alpha,
            linestyle=line_style,
        )
    )
    return visimage


def _change_color_brightness(color, brightness_factor):
    """
    Depending on the brightness_factor, gives a lighter or darker color i.e. a color with
    less or more saturation than the original color.

    Args:
        color: color of the polygon. Refer to `matplotlib.colors` for a full list of
            formats that are accepted.
        brightness_factor (float): a value in [-1.0, 1.0] range. A lightness factor of
            0 will correspond to no change, a factor in [-1.0, 0) range will result in
            a darker color and a factor in (0, 1.0] range will result in a lighter color.

    Returns:
        modified_color (tuple[double]): a tuple containing the RGB values of the
            modified color. Each value in the tuple is in the [0.0, 1.0] range.
    """
    assert brightness_factor >= -1.0 and brightness_factor <= 1.0
    color = mplc.to_rgb(color)
    polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
    modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
    modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
    modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
    modified_color = colorsys.hls_to_rgb(polygon_color[0], modified_lightness, polygon_color[2])
    return modified_color


class VisImage:
    def __init__(self, img, scale=1.0):
        """
        Args:
            img (ndarray): an RGB image of shape (H, W, 3).
            scale (float): scale the input image
        """
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(self, img):
        """
        Args:
            Same as in :meth:`__init__()`.

        Returns:
            fig (matplotlib.pyplot.figure): top level container for all the image plot elements.
            ax (matplotlib.pyplot.Axes): contains figure elements and sets the coordinate system.
        """
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        ax.set_xlim(0.0, self.width)
        ax.set_ylim(self.height)

        self.fig = fig
        self.ax = ax

    def save(self, filepath):
        """
        Args:
            filepath (str): a string that contains the absolute path, including the file name, where
                the visualized image will be saved.
        """
        if filepath.lower().endswith(".jpg") or filepath.lower().endswith(".png"):
            # faster than matplotlib's imshow
            cv2.imwrite(filepath, self.get_image()[:, :, ::-1])
        else:
            # support general formats (e.g. pdf)
            self.ax.imshow(self.img, interpolation="nearest")
            self.fig.savefig(filepath)

    def get_image(self):
        """
        Returns:
            ndarray: the visualized image of shape (H, W, 3) (RGB) in uint8 type.
              The shape is scaled w.r.t the input image using the given `scale` argument.
        """
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        if (self.width, self.height) != (width, height):
            img = cv2.resize(self.img, (width, height))
        else:
            img = self.img

        # buf = io.BytesIO()  # works for cairo backend
        # canvas.print_rgba(buf)
        # width, height = self.width, self.height
        # s = buf.getvalue()

        buffer = np.frombuffer(s, dtype="uint8")

        # imshow is slow. blend manually (still quite slow)
        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)

        try:
            import numexpr as ne  # fuse them with numexpr

            visualized_image = ne.evaluate("img * (1 - alpha / 255.0) + rgb * (alpha / 255.0)")
        except ImportError:
            alpha = alpha.astype("float32") / 255.0
            visualized_image = img * (1 - alpha) + rgb * alpha

        visualized_image = visualized_image.astype("uint8")

        return visualized_image


def show_annotation(filename, box, pre_box):
    print("box:", box)
    print("pre_box:", pre_box)
    img = cv2.imread(filename)
    # rectangle
    if box is not None:
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])), (0,0,255), 2)
    cv2.rectangle(img, (int(pre_box[0]), int(pre_box[1])), (int(pre_box[0] + pre_box[2]), int(pre_box[1] + pre_box[3])),
                  (255,0,0), 2)

    cv2.imwrite(filename, img)


def show_label_and_pre(filename, box_label, mask_label, box_pre, mask_pre, image_size=480):
    vimg = np.ones((image_size, image_size))*255
    cv2.rectangle(
        vimg, (int(box_label[0]), int(box_label[1])),
        (int(box_label[0] + box_label[2]), int(box_label[1] + box_label[3])),
        (0, 0, 255), 2)
    cv2.rectangle(
        vimg, (int(box_pre[0]), int(box_pre[1])),
        (int(box_pre[0] + box_pre[2]), int(box_pre[1] + box_pre[3])),
        (0, 0, 255), 2)
    cv2.imwrite(filename, vimg)


def cv2_multiline(img, polygon):
    length = int(len(polygon)/2)
    for i in range(1, length):
        point1 = polygon[2*i-2], polygon[2*i-1]
        point2 = polygon[2*i], polygon[2*i+1]
        cv2.line(img, point1, point2, (0,0,255), 1)
