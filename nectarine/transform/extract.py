from .transform import TransformLayer


class Extractor:
    def __init__(self, schema: dict):
        self.schema = schema

    def fit(self, x) -> TransformLayer:
        # todo:
        #   if feature is id -> apply ordinal encoder
        #   if feature is category -> one-hot encode it
        #   if feature is number -> get max/min
        #   if feature is text -> apply pre-trained model
        #   --> save results to config, return in TransformLayer
        return TransformLayer(self.schema)
