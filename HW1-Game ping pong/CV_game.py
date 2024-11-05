# ПИНГ-ПОНГ

import cv2
import numpy as np
import requests

# Параметры игры
width = 640  # ширина окна
height = 480  # высота окна
ball_radius = 20  # радиус мячика
paddle_width = 10  # ширина ракетки
paddle_height = 100  # высота ракетки
paddle_speed = 30  # скорость перемещения ракеток

# Положение мячика и его скорость
ball_pos = [width // 2, height // 2] # Начинаем с середины

# Скорость мячика
ball_speed = [3, 3]

# Положение ракеток (игрок 1 слева, игрок 2 справа)

paddle1_pos = [20, height // 2 - paddle_height // 2]
paddle2_pos = [width - 30, height // 2 - paddle_height // 2]

# Счет
score1, score2 = 0, 0

# Создание игрового окна
window_name = 'Ping-Pong'
cv2.namedWindow(window_name)

# Загрузка изображения стола для тенниса
url_table_tennis = 'https://github.com/aleksandrbogdanovmsu/comp_vision_classes/blob/main/HW1-Game%20ping%20pong/TennisTeble.jpg?raw=true'
response = requests.get(url_table_tennis)
table_tennis = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)

# Обрезка изображения стола
x1, x2 = 85, 920  # Координаты по ширине
y1, y2 = 130, 560  # Координаты по высоте

table_tennis = table_tennis[y1:y2, x1:x2]

# Изменение размера обрезанного изображения до размеров игрового окна
table_tennis = cv2.resize(table_tennis, (width, height))

# РАКЕТКИ

# Загрузка изображения ракетки
url_racket = 'https://github.com/aleksandrbogdanovmsu/comp_vision_classes/blob/main/HW1-Game%20ping%20pong/racket.jpg?raw=true'
response = requests.get(url_racket)
racket_img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_UNCHANGED)

# Убираем фон - достаем ракетку

# Преобразование изображения в формат с альфа-каналом (RGBA) для прозрачности

racket_img = cv2.cvtColor(racket_img, cv2.COLOR_BGR2BGRA)

# Удаление фона (в данном случае, можно предположить, что фон белый или прозрачный)

# Вытаскиваем альфа-канал и обрабатываем его
gray_racket = cv2.cvtColor(racket_img, cv2.COLOR_BGR2GRAY)


_, mask = cv2.threshold(gray_racket, 240, 255, cv2.THRESH_BINARY)
racket_img[mask == 255] = [0, 0, 0, 0]  # Прозрачность

# Функция удаления фона

def remove_background(image):
    # Преобразуем изображение в пространство HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Определим диапазон белого цвета в HSV
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([180, 25, 255], dtype=np.uint8)

    # Создаем маску для белого цвета
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Инвертируем маску, чтобы выделить ракетку
    mask_inv = cv2.bitwise_not(mask)

    # Применяем маску к изображению
    result = cv2.bitwise_and(image, image, mask=mask_inv)

    return result

# Удаляем фон
racket_no_bg = remove_background(racket_img)

# Изменение размера ракетки под игровое поле
resized_racket = cv2.resize(racket_no_bg, (paddle_width, paddle_height))

# Удаляем фон мячика и добавляем альфа-канал
def remove_background_with_alpha(image):
    # Преобразуем изображение в пространство HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Определим диапазон белого цвета в HSV
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([180, 25, 255], dtype=np.uint8)

    # Создаем маску для белого цвета
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Инвертируем маску, чтобы выделить мяч
    mask_inv = cv2.bitwise_not(mask)

    # Создаем новое изображение с альфа-каналом
    b, g, r = cv2.split(image)
    alpha = mask_inv  # Альфа-канал основан на маске

    # Объединяем цветовые каналы с альфа-каналом
    result = cv2.merge((b, g, r, alpha))

    return result

# Загрузка изображения мячика
url_ball = 'https://github.com/aleksandrbogdanovmsu/comp_vision_classes/blob/main/HW1-Game%20ping%20pong/ball.jpg?raw=true'
response = requests.get(url_ball)
ball_img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_UNCHANGED)

# Удаляем фон мячика и добавляем альфа-канал
ball_no_bg = remove_background_with_alpha(ball_img)

# Изменение размера мячика под игровые параметры
ball_no_bg_resized = cv2.resize(ball_no_bg, (ball_radius * 2, ball_radius * 2))

# Добавляем угол для поворота мячика
ball_angle = 0
rotation_speed = 5  # скорость вращения мячика


def rotate_image(image, angle):
    # Получаем размеры изображения
    h, w = image.shape[:2]

    # Определяем центр изображения
    center = (w // 2, h // 2)

    # Создаем матрицу поворота
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Поворачиваем изображение
    rotated_image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR)

    return rotated_image


def draw_game():
    global ball_angle  # используем глобальный угол вращения
    img = table_tennis.copy()

    # Разделяем изображение ракетки на цвет (RGB) и альфа-канал
    racket_rgb = resized_racket[:, :, :3]
    racket_alpha = resized_racket[:, :, 3]

    # Обработка для левой ракетки
    y1, y2 = paddle1_pos[1], paddle1_pos[1] + racket_rgb.shape[0]
    x1, x2 = paddle1_pos[0], paddle1_pos[0] + racket_rgb.shape[1]

    for c in range(3):
        img[y1:y2, x1:x2, c] = img[y1:y2, x1:x2, c] * (1 - racket_alpha / 255.0) + racket_rgb[:, :, c] * (
                    racket_alpha / 255.0)

    # Аналогично для правой ракетки
    y1, y2 = paddle2_pos[1], paddle2_pos[1] + racket_rgb.shape[0]
    x1, x2 = paddle2_pos[0], paddle2_pos[0] + racket_rgb.shape[1]

    for c in range(3):
        img[y1:y2, x1:x2, c] = img[y1:y2, x1:x2, c] * (1 - racket_alpha / 255.0) + racket_rgb[:, :, c] * (
                    racket_alpha / 255.0)

    # Рисуем мячик
    ball_center = (ball_pos[0], ball_pos[1])
    y1, y2 = ball_center[1] - ball_radius, ball_center[1] + ball_radius
    x1, x2 = ball_center[0] - ball_radius, ball_center[0] + ball_radius

    # Убедимся, что область мячика не выходит за границы изображения
    y1 = max(y1, 0)
    y2 = min(y2, height)
    x1 = max(x1, 0)
    x2 = min(x2, width)

    # Корректируем размеры мячика под выбранную область
    ball_resized = cv2.resize(ball_no_bg_resized, (x2 - x1, y2 - y1))

    # Вращаем мячик
    rotated_ball = rotate_image(ball_resized, ball_angle)

    # Наложение мячика с учетом прозрачности
    ball_alpha = rotated_ball[:, :, 3]
    for c in range(3):
        img[y1:y2, x1:x2, c] = img[y1:y2, x1:x2, c] * (1 - ball_alpha / 255.0) + rotated_ball[:, :, c] * (
                    ball_alpha / 255.0)

    # Отображаем счет
    cv2.putText(img, f"{score1}", (width // 4, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, f"{score2}", (3 * width // 4, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return img


while True:
    # Рисуем текущее состояние игры
    frame = draw_game()

    # Обрабатываем перемещение мячика
    ball_pos[0] += ball_speed[0]
    ball_pos[1] += ball_speed[1]

    # Обновляем угол вращения мячика
    ball_angle += rotation_speed
    ball_angle %= 360  # Обнуляем после 360 градусов

    # Проверка столкновения с верхней и нижней границами
    if ball_pos[1] - ball_radius <= 0 or ball_pos[1] + ball_radius >= height:
        ball_speed[1] = -ball_speed[1]

    # Проверка столкновения с ракетками
    if (paddle1_pos[0] <= ball_pos[0] - ball_radius <= paddle1_pos[0] + paddle_width and
            paddle1_pos[1] <= ball_pos[1] <= paddle1_pos[1] + paddle_height):
        ball_speed[0] = -ball_speed[0]

    if (paddle2_pos[0] <= ball_pos[0] + ball_radius <= paddle2_pos[0] + paddle_width and
            paddle2_pos[1] <= ball_pos[1] <= paddle2_pos[1] + paddle_height):
        ball_speed[0] = -ball_speed[0]

    # Проверка если мячик ушел за край (очки)
    if ball_pos[0] - ball_radius <= 0:
        score2 += 1
        ball_pos = [width // 2, height // 2]
        ball_speed = [3, 3]  # Сброс скорости

    if ball_pos[0] + ball_radius >= width:
        score1 += 1
        ball_pos = [width // 2, height // 2]
        ball_speed = [-3, 3]  # Сброс скорости

    # Проверка победы
    if score1 == 11 or score2 == 11:
        print("Игра окончена!")
        break

    # Обрабатываем перемещение ракеток
    key = cv2.waitKey(30) & 0xFF

    # Обработка перемещения левой ракетки
    if key == ord('w'):
        paddle1_pos[1] = max(paddle1_pos[1] - paddle_speed, 0)  # Ограничиваем верхнюю границу
    if key == ord('s'):
        paddle1_pos[1] = min(paddle1_pos[1] + paddle_speed, height - paddle_height)  # Ограничиваем нижнюю границу

    # Обработка перемещения правой ракетки
    if key == ord('i'):
        paddle2_pos[1] = max(paddle2_pos[1] - paddle_speed, 0)  # Ограничиваем верхнюю границу
    if key == ord('k'):
        paddle2_pos[1] = min(paddle2_pos[1] + paddle_speed, height - paddle_height)  # Ограничиваем нижнюю границу

    # Отображаем кадр игры
    cv2.imshow(window_name, frame)

    # Условие выхода
    if key == 27:  # Нажать Esc для выхода
        break

# Закрытие окна
cv2.destroyAllWindows()
