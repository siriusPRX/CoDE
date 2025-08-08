from .flip import horizontal_flip, vertical_flip
from .auc import cal_auc
from .f1 import cal_f1
from .iou import cal_iou
from .averagemeter import AverageMeter
from .logger import Logger
from .save_logs import save_logs
from .save_args import save_args
import math
import matplotlib.pyplot as plt

def display(func):
    def calculate_rows_columns(num_images):
        sqrt_num_images = math.sqrt(num_images)
        rows = int(sqrt_num_images)
        columns = int(sqrt_num_images)
        while rows * columns < num_images:
            columns += 1
        return rows, columns

    def wrapper(*args):
        display_name, result = func(*args)
        rows, columns = calculate_rows_columns(len(display_name))
        plt.figure(figsize=(10, 5))
        for i in range(len(display_name)):
            image = result[i]
            plt.subplot(rows, columns, i + 1)
            plt.imshow(image, cmap=None if len(result[i].shape) == 3 else 'gray'), plt.axis('off'), plt.title(f'{display_name[i]}')
        plt.tight_layout()
        plt.show()
        return result
    return wrapper

@display
def show_result(display_name, result):
    return display_name, result