import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
import time


import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
import time
import gc


def interpolate_images(img1, img2, alpha):
    # Assicurati che entrambe le immagini siano np.float32
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # Se le immagini non sono normalizzate (valori tra 0 e 1), normalizzale
    if img1.max() > 1.0:
        img1 /= 255.0

    return cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)

# seed diverso sempre
np.random.seed()

# Crea una finestra per mostrare il filmato
width = 960
heigh = 540
cv2.namedWindow('Generated Film', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Generated Film', width, heigh)
cv2.setWindowProperty('Generated Film', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # Set the window to fullscreen mode

# plotta logo
print('Carico il logo...')
logo_image = cv2.imread('esp_logo.png')
logo_image = cv2.rotate(logo_image, cv2.ROTATE_90_CLOCKWISE)
logo_image = cv2.resize(logo_image, (width*2, heigh*2), interpolation=cv2.INTER_LANCZOS4)
cv2.imshow('Generated Film', logo_image)
cv2.waitKey(1000)

# Carica il modello TFLite
interpreter = tflite.Interpreter(model_path="distill_quant_edgetpu.tflite",
                                 experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

images_list = []  # Step 1: lista per memorizzare le immagini
first_run = True # per mantenere l'immagine
transition_steps = 30  # Numero di passi nella transizione
transition_time = 0.5  # Tempo totale della transizione in secondi
step_time = transition_time / transition_steps  # Tempo per ogni passo della tr$

# noise_vector = torch.rand(1, 256, 1, 1)
noise_vector = np.random.rand(1, 256, 1, 1).astype(np.float32)

while True:
    # end_vector = torch.rand(1, 256, 1, 1)
    end_vector = np.random.rand(1, 256, 1, 1).astype(np.float32)
    num_passi = 30
    step = (end_vector - noise_vector) / num_passi

    for i in range(num_passi):
        start_time = time.perf_counter()

        # Setta l'input per l'Interpreter e invoca l'Interpreter
        # Converti l'input float32 in uint8
        noise_vector_quant = np.round(noise_vector * 255).astype(np.uint8)
        interpreter.set_tensor(input_details[0]['index'], noise_vector_quant)
        interpreter.invoke()

        # Ottieni i risultati dell'inferenza
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Converti l'output uint8 in float32
        output_data_float = output_data.astype(np.float32) / 255

        image = output_data_float.squeeze(0).transpose(1, 2, 0)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image = cv2.resize(image, (width, heigh), interpolation=cv2.INTER_LANCZOS4)

        # debug
        cv2.imshow('Generated Film', image)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

        images_list.append(image)

        # Mostra la prima immagine subito
        if i == 0:
            if first_run:
                print('First image created...')

                # Transizione dal logo alla prima immagine
                logo_image = cv2.resize(logo_image, (width, heigh), interpolation=cv2.INTER_LANCZOS4)
                for step in range(transition_steps):
                    transition_img = interpolate_images(logo_image, image, step / transition_steps)
                    transition_img_uint8 = np.clip(transition_img * 255, 0, 255).astype(np.uint8)  # Riporta l'immagine nell'intervallo di 8 bit e la c$
                    cv2.imshow('Generated Film', transition_img_uint8)
                    cv2.waitKey(int(step_time * 1000))  # Aspetta per il tempo $
                first_run = False
            else:
                cv2.imshow('Generated Film', last_image)
                cv2.waitKey(100)

        end_time = time.perf_counter()
        cpu_time = (end_time - start_time)
        print(f'Ciclo {i + 1}/{num_passi} e ci ha messo {round(cpu_time, 4)} secondi', end='\r')

        noise_vector += step

    noise_vector = end_vector

    # Mostra le immagini
    start_time = time.perf_counter()
    for img in images_list:
        cv2.imshow('Generated Film', img)
        cv2.waitKey(20)

    end_time = time.perf_counter()
    cpu_time = (end_time - start_time)
    print(f'transizione di {len(images_list)} in {round(cpu_time ,4)} secondi')


    last_image = img
    images_list = []  # pulisco il vettore transizione
    gc.collect()
    #
    #
    #
    #     # Mostra la prima immagine subito
    #     if i == 0:
    #         if first_run:
    #             print('First image created...')
    #             cv2.imshow('Generated Film', image)
    #             # if cv2.waitKey(1) & 0xFF == ord('q'):
    #             #     break
    #             first_run = False
    #         else:
    #             cv2.imshow('Generated Film', last_image)
    #             # if cv2.waitKey(1) & 0xFF == ord('q'):
    #             #     break
    #             # pass
    #
    #     end_time = time.perf_counter()
    #     cpu_time = (end_time - start_time)
    #     print(f'Ciclo {i+1}/{num_passi} e ci ha messo {round(cpu_time ,4)} secondi', end='\r')
    #
    #     noise_vector += step
    #
    # noise_vector = end_vector
    #
    # # Mostra le immagini
    # start_time = time.perf_counter()
    # for img in images_list:
    #     cv2.imshow('Generated Film', img)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # end_time = time.perf_counter()
    # cpu_time = (end_time - start_time)
    # print(f'transizione di {len(images_list)} in {round(cpu_time ,4)} secondi')
    #
    #
    # last_image = img
    # images_list = []  # pulisco il vettore transizione