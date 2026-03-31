import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, RNN, Input, Lambda, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

# ---------------------------------------------------------------------------
# Data augmentation (optional, applied at input)
# ---------------------------------------------------------------------------
def build_augmenter():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.3),
        tf.keras.layers.RandomZoom(0.3),
        tf.keras.layers.RandomContrast(0.3),
        tf.keras.layers.RandomBrightness(0.2)
    ], name="data_augmentation")


# ---------------------------------------------------------------------------
# Liquid Neural Network (LNN) cell – main reasoning model
# ---------------------------------------------------------------------------
class LiquidCell(tf.keras.layers.Layer):
    """
    Liquid Time-Constant (LTC) RNN cell for adaptive, continuous dynamics.
    Used as the main model on top of CNN features.
    """
    def __init__(self, units, **kwargs):
        super(LiquidCell, self).__init__(**kwargs)
        self.units = units
        self.state_size = units
        self.layer_norm = tf.keras.layers.LayerNormalization(name="lnn_lnorm")

    def build(self, input_shape):
        self.w_in = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            regularizer=l2(1e-4),
            name='w_in'
        )
        self.w_rec = self.add_weight(
            shape=(self.units, self.units),
            initializer='glorot_uniform',
            regularizer=l2(1e-4),
            name='w_rec'
        )
        self.b = self.add_weight(shape=(self.units,), initializer='zeros', name='b')
        self.tau = self.add_weight(
            shape=(self.units,),
            initializer='ones',
            name='tau',
            constraint=tf.keras.constraints.NonNeg()
        )
        self.layer_norm.build((None, self.units))
        self.built = True

    def call(self, inputs, states):
        prev_state = states[0]
        base_input = tf.matmul(inputs, self.w_in) + tf.matmul(prev_state, self.w_rec) + self.b
        liquid_tau = self.tau * tf.math.sigmoid(base_input)
        new_state = prev_state + (base_input - prev_state) * (liquid_tau + 1e-7)
        new_state = self.layer_norm(new_state)
        return new_state, [new_state]


# ---------------------------------------------------------------------------
# 1) CNN feature extractor (image → feature vector)
# ---------------------------------------------------------------------------
def build_cnn_feature_extractor(input_shape=(224, 224, 3), name="cnn_feature_extractor"):
    """
    CNN backbone for feature extraction only. No classification head.
    Returns: (model, last_conv_layer_name) for use in full model and Grad-CAM.
    """
    inputs = Input(shape=input_shape, name="image_input")
    base_cnn = ResNet50V2(weights='imagenet', include_top=False, input_tensor=inputs)
    base_cnn.trainable = False

    last_conv_layer = base_cnn.get_layer('post_relu')
    features = GlobalAveragePooling2D(name="cnn_gap")(last_conv_layer.output)
    features = BatchNormalization(name="cnn_bn")(features)
    features = Dropout(0.4, name="cnn_dropout")(features)

    cnn_model = Model(inputs=base_cnn.input, outputs=features, name=name)
    return cnn_model, last_conv_layer.name, base_cnn


# ---------------------------------------------------------------------------
# 2) Liquid Neural Network – main model (feature vector → class logits)
# ---------------------------------------------------------------------------
def build_lnn_classifier(feature_dim, num_classes=8, lnn_units=(384, 192), name="lnn_classifier"):
    """
    LNN head: takes CNN feature vector and outputs multi-label predictions.
    Uses a stack of LiquidCell layers as the main reasoning model.
    """
    inputs = Input(shape=(feature_dim,), name="feature_input")
    x = BatchNormalization(name="lnn_bn_in")(inputs)
    x = Dropout(0.45, name="lnn_drop_in")(x)

    # Main model: stack of Liquid (LTC) layers. Each RNN needs (batch, timesteps, features).
    for i, units in enumerate(lnn_units):
        x_seq = Lambda(lambda t: tf.expand_dims(t, axis=1), name=f"to_seq_{i}")(x)
        x = RNN(LiquidCell(units), name=f"liquid_layer_{i}")(x_seq)
        x = BatchNormalization(name=f"lnn_bn_{i}")(x)
        x = Dropout(0.5, name=f"lnn_drop_{i}")(x)

    outputs = Dense(
        num_classes,
        activation="sigmoid",
        name="disease_predictions",
        kernel_regularizer=l2(1e-4),
    )(x)
    return Model(inputs=inputs, outputs=outputs, name=name)


# ---------------------------------------------------------------------------
# Full model: CNN (feature extraction) → LNN (main model)
# ---------------------------------------------------------------------------
def build_concept_aware_lnn(input_shape=(224, 224, 3), num_classes=8):
    """
    Architecture:
    - CNN (ResNet50V2): feature extraction only.
    - LNN (LiquidCell stack): main model for classification.

    Returns: (full_model, last_conv_layer_name, base_cnn) for training and Grad-CAM.
    """
    inputs = Input(shape=input_shape)
    x = build_augmenter()(inputs)

    # 1) CNN feature extraction (single backbone in the graph)
    base_cnn = ResNet50V2(weights='imagenet', include_top=False, input_tensor=x)
    base_cnn.trainable = False
    last_conv = base_cnn.get_layer('post_relu')
    last_conv_layer_name = last_conv.name

    features = GlobalAveragePooling2D(name="cnn_gap")(last_conv.output)
    features = BatchNormalization(name="cnn_bn")(features)
    features = Dropout(0.4, name="cnn_dropout")(features)

    # 2) LNN main model (classification head). ResNet50V2 GAP output dim is 2048.
    feature_dim = int(features.shape[-1]) if features.shape[-1] is not None else 2048
    lnn_head = build_lnn_classifier(
        feature_dim=feature_dim,
        num_classes=num_classes,
        lnn_units=(384, 192),
        name="lnn_classifier",
    )
    outputs = lnn_head(features)

    full_model = Model(inputs=inputs, outputs=outputs, name="CNN_LNN_Concept_Aware")
    return full_model, last_conv_layer_name, base_cnn
