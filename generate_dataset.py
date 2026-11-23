import cv2
import numpy as np

from pathlib import Path


# config

BASE_DIR = Path(__file__).resolve().parent

ASSETS = BASE_DIR / "assets"
OUTPUT = BASE_DIR / "dataset"

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

CARD_WIDTH = 400
CARD_HEIGHT = 250

SPLITS = {
    "train": 2000,
    "val": 300
}

CARD_FILES = [
    "credencial_ficticia.png",
    "ine_ficticia.png",
    "licencia_ficticia.png"
]

SEED = 42
RNG = np.random.default_rng(SEED)

# Escala de la credencial
MIN_SCALE = 0.45
MAX_SCALE = 0.95

# Intensidad de perspectiva
PERSPECTIVE_JITTER = 35

# Margen minimo respecto a los bordes
MARGIN = 10

# Area minima del cuadrilatero
MIN_CARD_AREA = 5000

# Probabilidades de efectos
BLUR_PROBABILITY = 0.25
SHADOW_PROBABILITY = 0.35
NOISE_PROBABILITY = 0.40

# Eliminar archivos generados anteriormente
CLEAN_OUTPUT = True


# cargar plantillas

def load_cards():

    cards = []

    for filename in CARD_FILES:

        path = ASSETS / filename

        if not path.exists():
            raise FileNotFoundError(
                f"No se encontro: {path}"
            )

        # Mantener canal alfa si existe
        image = cv2.imread(
            str(path),
            cv2.IMREAD_UNCHANGED
        )

        if image is None:
            raise ValueError(
                f"No se pudo leer: {path}"
            )

        # Convertir a BGR
        if image.ndim == 2:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR
            )

        elif image.shape[2] == 4:

            alpha = (
                image[:, :, 3:4].astype(np.float32)
                / 255.0
            )

            white = np.full(
                image[:, :, :3].shape,
                255,
                dtype=np.float32
            )

            image = (
                image[:, :, :3].astype(np.float32)
                * alpha
                +
                white * (1.0 - alpha)
            ).astype(np.uint8)

        # Ajustar al tamaño deseado
        image = cv2.resize(
            image,
            (CARD_WIDTH, CARD_HEIGHT),
            interpolation=cv2.INTER_AREA
        )

        cards.append(image)

        print(f"Plantilla cargada: {filename}")

    return cards


# utilidades para fondos y texturas

def random_color(low=30, high=225):

    return RNG.integers(
        low,
        high,
        size=3
    ).astype(np.float32)


def add_fine_noise(image, intensity=8):

    noise = RNG.normal(
        0,
        intensity,
        image.shape
    )

    return np.clip(
        image.astype(np.float32) + noise,
        0,
        255
    ).astype(np.uint8)


def make_multiscale_noise():

    """
    Genera ruido en varias escalas para
    producir texturas menos uniformes.
    """

    h = IMAGE_HEIGHT
    w = IMAGE_WIDTH

    result = np.zeros(
        (h, w),
        dtype=np.float32
    )

    scales = [
        (8, 0.50),
        (32, 0.30),
        (128, 0.15),
        (256, 0.05)
    ]

    for size, weight in scales:

        small = RNG.random(
            (size, size)
        ).astype(np.float32)

        resized = cv2.resize(
            small,
            (w, h),
            interpolation=cv2.INTER_CUBIC
        )

        result += resized * weight

    result = cv2.normalize(
        result,
        None,
        0,
        1,
        cv2.NORM_MINMAX
    )

    return result


# ==========================================
# FONDO 1: MADERA
# ==========================================

def background_wood():

    h = IMAGE_HEIGHT
    w = IMAGE_WIDTH

    base = np.array(
        [90, 145, 185],
        dtype=np.float32
    )

    # Variar tonalidad
    base *= RNG.uniform(0.65, 1.20)

    y = np.arange(
        h,
        dtype=np.float32
    )

    frequency = RNG.uniform(
        0.025,
        0.065
    )

    phase = RNG.uniform(
        0,
        2 * np.pi
    )

    grain = (
        np.sin(y * frequency + phase) * 12
    )

    grain += (
        np.sin(y * frequency * 3.5) * 5
    )

    # Bandas irregulares
    coarse = RNG.normal(
        0,
        7,
        size=(max(2, h // 12), 1)
    ).astype(np.float32)

    coarse = cv2.resize(
        coarse,
        (w, h)
    )

    texture = (
        grain[:, None] + coarse
    )

    image = (
        base[None, None, :]
        + texture[:, :, None]
    )

    image = np.broadcast_to(
        image,
        (h, w, 3)
    ).copy()

    image = np.clip(
        image,
        0,
        255
    ).astype(np.uint8)

    return add_fine_noise(image, 5)


# ==========================================
# FONDO 2: PAPEL
# ==========================================

def background_paper():

    h = IMAGE_HEIGHT
    w = IMAGE_WIDTH

    base_colors = [
        [235, 235, 235],
        [220, 225, 235],
        [200, 210, 220],
        [245, 245, 245],
        [190, 205, 215]
    ]

    color = np.array(
        base_colors[
            int(RNG.integers(len(base_colors)))
        ],
        dtype=np.float32
    )

    image = np.full(
        (h, w, 3),
        color,
        dtype=np.float32
    )

    texture = make_multiscale_noise()

    image += (
        (texture[:, :, None] - 0.5) * 25
    )

    return add_fine_noise(
        np.clip(image, 0, 255).astype(np.uint8),
        4
    )


# ==========================================
# FONDO 3: TELA
# ==========================================

def background_fabric():

    h = IMAGE_HEIGHT
    w = IMAGE_WIDTH

    color = random_color(45, 185)

    image = np.full(
        (h, w, 3),
        color,
        dtype=np.float32
    )

    x = np.arange(
        w,
        dtype=np.float32
    )

    y = np.arange(
        h,
        dtype=np.float32
    )

    spacing = int(
        RNG.integers(3, 8)
    )

    vertical = (
        np.sin(x * 2 * np.pi / spacing)
        * 7
    )

    horizontal = (
        np.sin(y * 2 * np.pi / spacing)
        * 7
    )

    pattern = (
        horizontal[:, None]
        +
        vertical[None, :]
    )

    image += pattern[:, :, None]

    image = np.clip(
        image,
        0,
        255
    ).astype(np.uint8)

    return add_fine_noise(image, 6)


# ==========================================
# FONDO 4: MARMOL / PIEDRA
# ==========================================

def background_marble():

    h = IMAGE_HEIGHT
    w = IMAGE_WIDTH

    color = random_color(80, 215)

    texture = make_multiscale_noise()

    yy, xx = np.mgrid[
        0:h,
        0:w
    ]

    frequency = RNG.uniform(
        0.015,
        0.035
    )

    veins = np.sin(
        xx * frequency
        +
        yy * frequency * 0.6
        +
        texture * 12
    )

    veins = veins * 25

    image = (
        color[None, None, :]
        +
        veins[:, :, None]
    )

    image = np.clip(
        image,
        0,
        255
    ).astype(np.uint8)

    return add_fine_noise(image, 5)


# ==========================================
# FONDO 5: SUPERFICIE GEOMETRICA
# ==========================================

def background_geometric():

    h = IMAGE_HEIGHT
    w = IMAGE_WIDTH

    image = np.full(
        (h, w, 3),
        random_color(35, 210),
        dtype=np.uint8
    )

    # Rectangulos y poligonos
    for _ in range(
        int(RNG.integers(5, 20))
    ):

        color = tuple(
            int(v) for v in random_color()
        )

        x1 = int(RNG.integers(0, w))
        y1 = int(RNG.integers(0, h))

        x2 = int(RNG.integers(0, w))
        y2 = int(RNG.integers(0, h))

        if RNG.random() < 0.5:

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                -1
            )

        else:

            radius = int(
                RNG.integers(10, 100)
            )

            cv2.circle(
                image,
                (x1, y1),
                radius,
                color,
                -1
            )

    image = cv2.GaussianBlur(
        image,
        (9, 9),
        0
    )

    return add_fine_noise(image, 5)


# ==========================================
# FONDO 6: DEGRADADO CON TEXTURA
# ==========================================

def background_gradient():

    h = IMAGE_HEIGHT
    w = IMAGE_WIDTH

    color1 = random_color()
    color2 = random_color()

    angle = RNG.uniform(
        0,
        2 * np.pi
    )

    yy, xx = np.mgrid[
        0:h,
        0:w
    ]

    projection = (
        xx * np.cos(angle)
        +
        yy * np.sin(angle)
    )

    minimum = projection.min()
    maximum = projection.max()

    t = (
        (projection - minimum)
        /
        (maximum - minimum + 1e-6)
    )

    image = (
        color1[None, None, :] * (1 - t[:, :, None])
        +
        color2[None, None, :] * t[:, :, None]
    )

    texture = make_multiscale_noise()

    image += (
        texture[:, :, None] - 0.5
    ) * 30

    image = np.clip(
        image,
        0,
        255
    ).astype(np.uint8)

    return add_fine_noise(image, 6)


# ==========================================
# SELECCION ALEATORIA DE FONDO
# ==========================================

BACKGROUND_FUNCTIONS = [
    background_wood,
    background_paper,
    background_fabric,
    background_marble,
    background_geometric,
    background_gradient
]


def create_background():

    index = int(
        RNG.integers(
            0,
            len(BACKGROUND_FUNCTIONS)
        )
    )

    background = BACKGROUND_FUNCTIONS[index]()

    return background


# ==========================================
# TRANSFORMAR CREDENCIAL
# ==========================================

def transform_card(card, background):

    h, w = card.shape[:2]

    source = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ])

    # Escala
    scale = RNG.uniform(
        MIN_SCALE,
        MAX_SCALE
    )

    # Rotacion completa
    angle = RNG.uniform(
        0,
        2 * np.pi
    )

    rotation = np.array([
        [
            np.cos(angle),
            -np.sin(angle)
        ],
        [
            np.sin(angle),
            np.cos(angle)
        ]
    ], dtype=np.float32)

    # Centrar coordenadas
    centered = (
        source - source.mean(axis=0)
    )

    # Rotar
    rotated = (
        centered @ rotation.T
    )

    rotated *= scale

    # Distorsion de perspectiva
    jitter = RNG.uniform(
        -PERSPECTIVE_JITTER,
        PERSPECTIVE_JITTER,
        size=(4, 2)
    ).astype(np.float32)

    destination = rotated + jitter

    # Verificar geometria inicial
    contour = destination.reshape(
        -1,
        1,
        2
    )

    if not cv2.isContourConvex(contour):
        return None

    if cv2.contourArea(contour) < MIN_CARD_AREA:
        return None

    # Obtener limites
    minimum = destination.min(axis=0)
    maximum = destination.max(axis=0)

    card_width = (
        maximum[0] - minimum[0]
    )

    card_height = (
        maximum[1] - minimum[1]
    )

    if card_width > IMAGE_WIDTH - 2 * MARGIN:
        return None

    if card_height > IMAGE_HEIGHT - 2 * MARGIN:
        return None

    # Posicion aleatoria dentro del fotograma
    shift_x = RNG.uniform(
        MARGIN - minimum[0],
        IMAGE_WIDTH - MARGIN - maximum[0]
    )

    shift_y = RNG.uniform(
        MARGIN - minimum[1],
        IMAGE_HEIGHT - MARGIN - maximum[1]
    )

    destination += np.array(
        [shift_x, shift_y],
        dtype=np.float32
    )

    # Homografia
    matrix = cv2.getPerspectiveTransform(
        source,
        destination
    )

    # Transformar tarjeta
    warped_card = cv2.warpPerspective(
        card,
        matrix,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    # Mascara original
    mask = np.full(
        (h, w),
        255,
        dtype=np.uint8
    )

    # Transformar mascara
    warped_mask = cv2.warpPerspective(
        mask,
        matrix,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    alpha = (
        warped_mask.astype(np.float32)
        / 255.0
    )[:, :, None]

    # Combinar con fondo
    image = (
        warped_card.astype(np.float32) * alpha
        +
        background.astype(np.float32) * (1 - alpha)
    )

    image = np.clip(
        image,
        0,
        255
    ).astype(np.uint8)

    return image, destination


# ==========================================
# SOMBRAS ALEATORIAS
# ==========================================

def add_shadow(image):

    h, w = image.shape[:2]

    # Campo de sombra suave
    shadow = np.zeros(
        (h, w),
        dtype=np.float32
    )

    # Poligono de sombra
    points = np.array([
        [
            int(RNG.integers(0, w)),
            int(RNG.integers(0, h))
        ]
        for _ in range(4)
    ], dtype=np.int32)

    hull = cv2.convexHull(points)

    cv2.fillConvexPoly(
        shadow,
        hull,
        1.0
    )

    # Suavizar bordes
    shadow = cv2.GaussianBlur(
        shadow,
        (0, 0),
        sigmaX=35
    )

    intensity = RNG.uniform(
        0.15,
        0.45
    )

    factor = (
        1.0 - shadow * intensity
    )

    result = (
        image.astype(np.float32)
        *
        factor[:, :, None]
    )

    return np.clip(
        result,
        0,
        255
    ).astype(np.uint8)


# ==========================================
# AUMENTO FOTOMETRICO
# ==========================================

def augment_image(image):

    # Brillo y contraste
    alpha = RNG.uniform(
        0.75,
        1.25
    )

    beta = RNG.uniform(
        -25,
        25
    )

    image = np.clip(
        image.astype(np.float32) * alpha + beta,
        0,
        255
    ).astype(np.uint8)

    # Sombras
    if RNG.random() < SHADOW_PROBABILITY:

        image = add_shadow(image)

    # Ruido
    if RNG.random() < NOISE_PROBABILITY:

        intensity = RNG.uniform(
            2,
            12
        )

        image = add_fine_noise(
            image,
            intensity
        )

    # Desenfoque
    if RNG.random() < BLUR_PROBABILITY:

        kernel = int(
            RNG.choice([3, 5])
        )

        image = cv2.GaussianBlur(
            image,
            (kernel, kernel),
            0
        )

    return image


# ==========================================
# CREAR ETIQUETA YOLO-POSE
# ==========================================

def create_label(corners):

    x = corners[:, 0]
    y = corners[:, 1]

    xmin = float(x.min())
    xmax = float(x.max())

    ymin = float(y.min())
    ymax = float(y.max())

    # Bounding box normalizada
    xc = (
        (xmin + xmax) / 2
    ) / IMAGE_WIDTH

    yc = (
        (ymin + ymax) / 2
    ) / IMAGE_HEIGHT

    bw = (
        xmax - xmin
    ) / IMAGE_WIDTH

    bh = (
        ymax - ymin
    ) / IMAGE_HEIGHT

    values = [
        0,
        xc,
        yc,
        bw,
        bh
    ]

    # Cuatro keypoints
    for px, py in corners:

        values.extend([
            float(px) / IMAGE_WIDTH,
            float(py) / IMAGE_HEIGHT,
            2
        ])

    # Formato YOLO
    label = (
        str(int(values[0]))
        +
        " "
        +
        " ".join(
            f"{value:.6f}"
            for value in values[1:]
        )
    )

    return label


# ==========================================
# CREAR DATA.YAML
# ==========================================

def create_yaml():

    yaml_path = OUTPUT / "data.yaml"

    content = f"""path: {OUTPUT.as_posix()}

train: images/train
val: images/val

kpt_shape: [4, 3]

flip_idx: [0, 1, 2, 3]

names:
  0: credential
"""

    yaml_path.write_text(
        content,
        encoding="utf-8"
    )

    print(f"Archivo creado: {yaml_path}")


# ==========================================
# LIMPIAR DATASET ANTERIOR
# ==========================================

def clean_dataset():

    if not CLEAN_OUTPUT:
        return

    for split in SPLITS:

        for folder_name, extension in [
            ("images", ".jpg"),
            ("labels", ".txt")
        ]:

            folder = (
                OUTPUT
                / folder_name
                / split
            )

            if not folder.exists():
                continue

            # Solo eliminar archivos generados
            for file in folder.glob(
                f"card_*{extension}"
            ):

                file.unlink()

    print("Dataset anterior limpiado.")


# ==========================================
# GENERAR DATASET
# ==========================================

def generate_dataset():

    print("\nCargando plantillas...\n")

    cards = load_cards()

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    clean_dataset()

    print("\nGenerando dataset...\n")

    for split, quantity in SPLITS.items():

        image_dir = (
            OUTPUT / "images" / split
        )

        label_dir = (
            OUTPUT / "labels" / split
        )

        image_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        label_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        generated = 0
        attempts = 0

        # Cantidad maxima de intentos
        max_attempts = quantity * 30

        while generated < quantity:

            attempts += 1

            if attempts > max_attempts:

                raise RuntimeError(
                    "No fue posible generar todas "
                    "las imagenes. Revisa los "
                    "parametros geometricos."
                )

            # Seleccionar tarjeta
            card_index = int(
                RNG.integers(
                    0,
                    len(cards)
                )
            )

            card = cards[card_index]

            # Generar fondo
            background = create_background()

            # Transformar tarjeta
            result = transform_card(
                card,
                background
            )

            if result is None:
                continue

            image, corners = result

            # Aumentos visuales
            image = augment_image(image)

            # Nombre de archivo
            filename = (
                f"card_{generated:05d}"
            )

            image_path = (
                image_dir
                / f"{filename}.jpg"
            )

            label_path = (
                label_dir
                / f"{filename}.txt"
            )

            # Guardar imagen
            success = cv2.imwrite(
                str(image_path),
                image,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    95
                ]
            )

            if not success:
                raise IOError(
                    f"No se pudo guardar {image_path}"
                )

            # Crear etiqueta
            label = create_label(corners)

            # Guardar etiqueta
            label_path.write_text(
                label + "\n",
                encoding="utf-8"
            )

            generated += 1

            # Mostrar progreso
            if generated % 100 == 0:

                print(
                    f"{split}: "
                    f"{generated}/{quantity}"
                )

        print(
            f"\n{split} completado: "
            f"{generated} imagenes\n"
        )

    # Crear configuracion YOLO
    create_yaml()

    print("\nDataset generado correctamente.")

    print(
        f"Total: {sum(SPLITS.values())} imagenes"
    )

    print(f"Ubicacion: {OUTPUT}")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    generate_dataset()