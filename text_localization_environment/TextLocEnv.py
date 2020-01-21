import gym
from gym import spaces
from gym.utils import seeding
from chainer.backends import cuda
from PIL import Image, ImageDraw
from PIL.Image import LANCZOS, MAX_IMAGE_PIXELS
import numpy as np
from text_localization_environment.ImageMasker import ImageMasker
import copy


class TextLocEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array', 'box']}

    DURATION_PENALTY = 0.03
    HISTORY_LENGTH = 10
    # ⍺: factor relative to the current box size that is used for every transformation action
    ALPHA = 0.2
    # η: Reward of the trigger action
    ETA = 70.0

    def __init__(self, image_paths, true_bboxes, gpu_id=-1, reward_function='single'):
        """
        :param image_paths: The paths to the individual images
        :param true_bboxes: The true bounding boxes for each image
        :param gpu_id: The ID of the GPU to be used. -1 if CPU should be used instead
        :type image_paths: String or list
        :type true_bboxes: numpy.ndarray
        :type gpu_id: int
        """

        # initialize self
        self.done = False

        self.action_space = spaces.Discrete(9)
        self.action_set = {0: self.right,
                           1: self.left,
                           2: self.up,
                           3: self.down,
                           4: self.bigger,
                           5: self.smaller,
                           6: self.fatter,
                           7: self.taller,
                           8: self.trigger
                           }
        # 224*224*4 (RGBA image) + 9 * 10 (on-hot-enconded history) = 200794
        self.observation_space = spaces.Tuple([spaces.Box(low=0, high=256, shape=(224,224,4)), spaces.Box(low=0,high=1,shape=(10,9))])
        self.gpu_id = gpu_id
        if self.gpu_id != -1:
            cuda.Device(self.gpu_id).use() # define gpu id fix for all later operations

        self.reward_function = reward_function

        if type(image_paths) is not list:
            image_paths = [image_paths]
        self.image_paths = image_paths
        self.true_bboxes = true_bboxes

        self.np_random, _ = seeding.np_random(None)

        self.episode_image = Image.new("RGBA", (224, 224))
        self.mask_array = np.full([224, 224, 1], 255, dtype=np.int32)
        self.reset()

    def step(self, action):
        """Execute an action and return
            state - the next state,
            reward - the reward,
            done - whether a terminal state was reached,
            info - any additional info"""
        assert self.action_space.contains(action), "%r (%s) is an invalid action" % (action, type(action))

        self.current_step += 1

        self.action_set[action]()

        reward = self.calculate_reward(action)

        self.history.insert(0, self.to_one_hot(action))
        self.history.pop()

        self.state = self.compute_state()

        return self.state, reward, self.done, {}

    def calculate_reward(self, action):
        reward = 0

        if self.action_set[action] == self.trigger:
            self.iou = self.compute_best_iou(self.episode_not_found_bboxes)

            if self.reward_function == "single":
                iou_with_ior = self.compute_best_iou(self.episode_found_bboxes)
                reward = self.ETA * (self.iou - (iou_with_ior**2)) - (self.current_step * self.DURATION_PENALTY)

            elif self.reward_function == "sum":
                sum_ior_ious = 0
                for ior_box in self.episode_found_bboxes:
                    sum_ior_ious += self.compute_iou(ior_box)
                reward = self.ETA * (self.iou - (sum_ior_ious**2)) - (self.current_step * self.DURATION_PENALTY)

        return reward

    def create_empty_history(self):
        flat_history = np.repeat([False], self.HISTORY_LENGTH * self.action_space.n)
        history = flat_history.reshape((self.HISTORY_LENGTH, self.action_space.n))

        return history.tolist()

    @staticmethod
    def to_four_corners_array(two_bbox):
        """
        Creates an array of bounding boxes with four corners out of a bounding box with two corners, so
        that the ImageMasker can be applied.

        :param two_bbox: Bounding box with two points, top left and bottom right

        :return: An array of bounding boxes that corresponds to the requirements of the ImageMasker
        """
        top_left = np.array([two_bbox[0], two_bbox[1]], dtype=np.int32)
        bottom_left = np.array([two_bbox[0], two_bbox[3]], dtype=np.int32)
        top_right = np.array([two_bbox[2], two_bbox[1]], dtype=np.int32)
        bottom_right = np.array([two_bbox[2], two_bbox[3]], dtype=np.int32)

        four_bbox = np.array([bottom_right, bottom_left, top_left, top_right])

        return np.array([four_bbox, four_bbox, four_bbox])

    def create_ior_mark(self, bbox):
        """
        Creates an IoR (inhibition of return) mark on the current image that crosses out the given bounding box.
        This is necessary to find multiple objects within one image
         :param bbox: Bounding box with two points, top left and bottom right
        """
        masker = ImageMasker(0)

        bbox = [bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]]

        center_height = round((bbox[3] + bbox[1]) / 2)
        center_width = round((bbox[2] + bbox[0]) / 2)
        height_frac = round((bbox[3] - bbox[1]) / 12)
        width_frac = round((bbox[2] - bbox[0]) / 12)

        horizontal_box = [bbox[0], center_height - height_frac, bbox[2], center_height + height_frac]
        vertical_box = [center_width - width_frac, bbox[1], center_width + width_frac, bbox[3]]

        horizontal_box_four_corners = self.to_four_corners_array(horizontal_box)
        vertical_box_four_corners = self.to_four_corners_array(vertical_box)

        array_module = np

        if self.gpu_id != -1:
            array_module = cuda.cupy
            horizontal_box_four_corners = cuda.to_gpu(horizontal_box_four_corners, self.gpu_id)
            vertical_box_four_corners = cuda.to_gpu(vertical_box_four_corners, self.gpu_id)

        self.mask_array = array_module.array(self.mask_array, dtype=np.int32)
        self.mask_array = masker.mask_array(self.mask_array, horizontal_box_four_corners, array_module)
        self.mask_array = masker.mask_array(self.mask_array, vertical_box_four_corners, array_module)

        if self.gpu_id != -1:
            to_cpu_mask = cuda.to_cpu(self.mask_array).astype(np.uint8)
            to_cpu_mask = np.squeeze(to_cpu_mask, axis=(2,))
            mask_image = Image.fromarray(to_cpu_mask, mode="L")
        else:
            self.mask_array = np.squeeze(self.mask_array, axis=(2,))
            mask_image = Image.fromarray(self.mask_array.astype(np.uint8))

        self.episode_image.putalpha(mask_image)

    def current_best_bbox(self, bboxes):
        best_bbox = None
        max_iou = 0
        for box in bboxes:
            current_iou = self.compute_iou(box)
            max_iou = max(max_iou, current_iou)
            if max_iou == current_iou:
                best_bbox = box

        return best_bbox

    def compute_best_iou(self, bboxes):
        max_iou = 0
        for box in bboxes:
            max_iou = max(max_iou, self.compute_iou(box))

        return max_iou

    def compute_iou(self, other_bbox):
        """Computes the intersection over union of the argument and the current bounding box."""
        intersection = self.compute_intersection(other_bbox)

        area_1 = (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
        area_2 = (other_bbox[1][0] - other_bbox[0][0]) * (other_bbox[1][1] - other_bbox[0][1])
        union = area_1 + area_2 - intersection

        return intersection / union

    def compute_intersection(self, other_bbox):
        left = max(self.bbox[0], other_bbox[0][0])
        top = max(self.bbox[1], other_bbox[0][1])
        right = min(self.bbox[2], other_bbox[1][0])
        bottom = min(self.bbox[3], other_bbox[1][1])

        if right < left or bottom < top:
            return 0

        return (right - left) * (bottom - top)

    def up(self):
        self.adjust_bbox(np.array([0, -1, 0, -1]))

    def down(self):
        self.adjust_bbox(np.array([0, 1, 0, 1]))

    def left(self):
        self.adjust_bbox(np.array([-1, 0, -1, 0]))

    def right(self):
        self.adjust_bbox(np.array([1, 0, 1, 0]))

    def bigger(self):
        self.adjust_bbox(np.array([-0.5, -0.5, 0.5, 0.5]))

    def smaller(self):
        self.adjust_bbox(np.array([0.5, 0.5, -0.5, -0.5]))

    def fatter(self):
        self.adjust_bbox(np.array([0, 0.5, 0, -0.5]))

    def taller(self):
        self.adjust_bbox(np.array([0.5, 0, -0.5, 0]))

    def trigger(self):
        self.done = True

    @staticmethod
    def box_size(box):
        width = box[2] - box[0]
        height = box[3] - box[1]

        return width * height

    def adjust_bbox(self, directions):
        ah = round(self.ALPHA * (self.bbox[3] - self.bbox[1]))
        aw = round(self.ALPHA * (self.bbox[2] - self.bbox[0]))

        adjustments = np.array([aw, ah, aw, ah])
        delta = directions * adjustments

        new_box = self.bbox + delta

        if self.box_size(new_box) < MAX_IMAGE_PIXELS:
            self.bbox = new_box

    def reset(self, image_index=None, stay_on_image=False, start_bbox=None, add_random_iors=True):
        """Reset the environment to its initial state (the bounding box covers the entire image"""
        print("begin reset")
        if not stay_on_image:
            self.history = self.create_empty_history()
            self.episode_image.close()

            if image_index is not None:
                self.episode_image = Image.open(self.image_paths[image_index])
                self.episode_true_bboxes = self.true_bboxes[image_index]
            else:
                random_index = self.np_random.randint(len(self.image_paths))
                self.episode_image = Image.open(self.image_paths[random_index])
                self.episode_true_bboxes = self.true_bboxes[random_index]

            self.mask_array = np.full([224, 224, 1], 255, dtype=np.int32)
            self.episode_found_bboxes = []
            self.episode_not_found_bboxes = copy.copy(self.episode_true_bboxes)
        else:
            found_word_bbox = self.current_best_bbox(self.episode_not_found_bboxes)
            if found_word_bbox:
                self.episode_found_bboxes.append(found_word_bbox)
                self.episode_not_found_bboxes.remove(found_word_bbox)
            if self.done:
                self.create_ior_mark(((self.bbox[0], self.bbox[1]), (self.bbox[2], self.bbox[3])))
            

        if self.episode_image.mode != 'RGBA':
            self.episode_image = self.episode_image.convert('RGBA')

        if start_bbox is None:
            self.bbox = np.array([0, 0, self.episode_image.width, self.episode_image.height])
        else:
            self.bbox = start_bbox

        self.current_step = 0
        self.state = self.compute_state()
        self.done = False

        if add_random_iors:
            self.add_random_iors()

        print("finished reset")
        return self.state

    def add_random_iors(self):
        """Add a random number of IoRs for correctly found bounding boxes"""
        self.episode_found_bboxes = []
        self.episode_not_found_bboxes = []
        number_of_iors = self.np_random.randint(len(self.episode_true_bboxes))
        for bbox_index in range(len(self.episode_true_bboxes)):
            bbox = self.episode_true_bboxes[bbox_index]
            if bbox_index < number_of_iors:
                self.create_ior_mark(bbox)
                self.episode_found_bboxes.append(bbox)
            else:
                self.episode_not_found_bboxes.append(bbox)

    def render(self, mode='human', return_as_file=False):
        """Render the current state"""

        if mode == 'human':
            copy = self.episode_image.copy()
            draw = ImageDraw.Draw(copy)
            draw.rectangle(self.bbox.tolist(), outline=(255, 255, 255, 255))
            if return_as_file:
                return copy
            copy.show()
            copy.close()
        elif mode is 'box':
            warped = self.get_warped_bbox_contents()
            if return_as_file:
                return warped
            warped.show()
            warped.close()
        elif mode is 'rgb_array':
            copy = self.episode_image.copy()
            draw = ImageDraw.Draw(copy)
            draw.rectangle(self.bbox.tolist(), outline=(255, 255, 255, 255))
            return np.array(copy)
        else:
            super(TextLocEnv, self).render(mode=mode)

    def get_warped_bbox_contents(self):
        cropped = self.episode_image.crop(self.bbox)
        return cropped.resize((224, 224), LANCZOS)

    def compute_state(self):
        warped = self.get_warped_bbox_contents()
        return np.array(warped, dtype=np.float32), np.array(self.history)

    def to_one_hot(self, action):
        line = np.zeros(self.action_space.n, np.bool)
        line[action] = 1

        return line