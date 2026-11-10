import typing as t
import os
import numpy as np
from types import SimpleNamespace

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


class AudioAttribute(AbstractLoader):
    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace, logger: object):
        self.logger = logger
        self.audio_feature_folder_path = getattr(ns, "audio_features", None)

        self.item_mapping = {}
        self.audio_features_shape = None

        #items = set(str(it) for it in items)
        inner_items = self.check_items_in_folder()

        self.users = users
        self.items = items & inner_items
        # built here too, not just in filter(): the coordinator only calls filter()
        # when intersecting with this set actually narrows it, so if items is already
        # a subset of inner_items, filter() never runs and item_mapping would
        # otherwise stay empty, leaving get_all_audio_features()'s output array
        # uninitialized.
        self.item_mapping = {int(item): val for val, item in enumerate(self.items)}

    def get_mapped(self) -> t.Tuple[t.Set[int], t.Set[int]]:
        return self.users, self.items

    def filter(self, users: t.Set[int], items: t.Set[int]):
        self.users = self.users & users
        self.items = self.items & items
        self.item_mapping = {int(item): val for val, item in enumerate(self.items)}

    def create_namespace(self) -> SimpleNamespace:
        ns = SimpleNamespace()
        ns.__name__ = "AudioAttribute"
        ns.object = self
        ns.audio_feature_folder_path = self.audio_feature_folder_path

        ns.item_mapping = self.item_mapping

        ns.audio_features_shape = self.audio_features_shape

        return ns

    def check_items_in_folder(self):
        items = set()
        if self.audio_feature_folder_path:
            items_folder = os.listdir(self.audio_feature_folder_path)
            items = items.union(set([int(f.split('.')[0]) for f in items_folder]))
            self.audio_features_shape = np.load(os.path.join(self.audio_feature_folder_path,
                                                             items_folder[0])).shape[0]
        return items

    def get_all_features(self, item_order: t.Optional[t.List[int]] = None):
        return self.get_all_audio_features(item_order)

    def get_all_audio_features(self, item_order: t.Optional[t.List[int]] = None):
        # item_order, when given, is the raw item id for each row the caller wants
        # (e.g. the dataset's private-item order) -- loading directly by id this way
        # avoids depending on this loader's own item_mapping enumeration order lining
        # up with whatever order the caller actually needs.
        items = item_order if item_order is not None else self.items
        all_features = np.empty((len(items), self.audio_features_shape))
        if self.audio_feature_folder_path:
            for row, item in enumerate(items):
                all_features[row] = np.load(self.audio_feature_folder_path + '/' + str(item) + '.npy')
        return all_features
