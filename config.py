# @File: config.py
import os

class UNetConfig:
    def __init__(
        self,
        # -----------------------
        # 训练参数
        # -----------------------
        epochs=200,
        batch_size=4,
        validation=10.0,
        out_threshold=0.5,
        optimizer='Adam',
        lr=1e-4,
        lr_decay_milestones=[20,50],
        lr_decay_gamma=0.9,
        weight_decay=1e-3,
        momentum=0.9,
        nesterov=True,

        # -----------------------
        # 模型参数
        # -----------------------
        model="U2Net",        # "U2Net'" / "UNet" / "NestedUNet"
        n_channels=3,
        n_classes=5,
        scale=1,
        fixed_size=(512,512),
        bilinear=True,
        deepsupervision=True,

        # -----------------------
        # 注意力机制（支持消融实验）
        # -----------------------
        attention_type="None",
        # 分层注意力，可覆盖全局注意力
        layer_attentions=None,
        # 消融实验：完全禁用注意力
        disable_attention=False
    ):
        super().__init__()

        # -----------------------
        # 数据与保存路径
        # -----------------------
        self.images_dir = './data/images'
        self.masks_dir = './data/masks'

        # 决定 checkpoint 文件夹名称
        if layer_attentions:
            attn_name = "custom"
        elif disable_attention:
            attn_name = "noatt"
        else:
            attn_name = str(attention_type)
        self.checkpoints_dir = os.path.join("./data/checkpoints", f"{model}_{attn_name}")
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        self.log_file = os.path.join(self.checkpoints_dir, f"{model}_{attn_name}_log.csv")

        # 加载与保存
        self.load = None  # ✅ 默认不加载权重，可设为 "checkpoints/model.pth"
        self.resume = False  # 是否从断点恢复训练

        # 设备与日志
        self.device = "cuda"
        self.num_workers = 4
        self.log_dir = "./runs"

        # -----------------------
        # 训练参数
        # -----------------------
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation = validation
        self.out_threshold = out_threshold
        self.optimizer = optimizer
        self.lr = lr
        self.lr_decay_milestones = lr_decay_milestones
        self.lr_decay_gamma = lr_decay_gamma
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov

        # -----------------------
        # 模型结构参数
        # -----------------------
        self.model = model
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.scale = scale
        self.fixed_size = fixed_size
        self.bilinear = bilinear
        self.deepsupervision = deepsupervision

        # -----------------------
        # 注意力机制
        # -----------------------
        self.attention_type = attention_type
        self.layer_attentions = layer_attentions or {
            "stage1": "se",
            "stage2": "se",
            "stage3": "cbam",
            "stage4": "cbam",
            "stage5": "eca",
            "stage6": "eca",
            "stage5d": "eca",
            "stage4d": "cbam",
            "stage3d": "se",
            "stage2d": "se",
            "stage1d": "None"
        }
        self.disable_attention = disable_attention
