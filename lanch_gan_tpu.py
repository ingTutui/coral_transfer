import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
import time



np.random.seed()
# Crea una finestra per mostrare il filmato
width = 512
heigh = 660
cv2.namedWindow('Generated Film', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Generated Film', width, heigh)


# Carica il modello TFLite

interpreter = tflite.Interpreter(model_path="distill_quant_edgetpu.tflite",
                                 experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

images_list = []  # Step 1: lista per memorizzare le immagini
first_run = True # per mantenere l'immagine

# noise_vector = torch.rand(1, 256, 1, 1)
noise_vector = np.random.rand(1, 256, 1, 1).astype(np.float32)

while True:
    # end_vector = torch.rand(1, 256, 1, 1)
    end_vector = np.random.rand(1, 256, 1, 1).astype(np.float32)
    num_passi = 5
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
        image = cv2.resize(image, (width, heigh))

        images_list.append(image)

        # Mostra la prima immagine subito
        if i == 0:
            if first_run:
                cv2.imshow('Generated Film', image)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
                first_run = False
            else:
                cv2.imshow('Generated Film', last_image)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
                # pass

        end_time = time.perf_counter()
        cpu_time = (end_time - start_time)
        print(f'Ciclo ci ha messo {round(cpu_time ,4)} secondi', end='\r')

        noise_vector += step

    noise_vector = end_vector

    # Mostra le immagini
    for img in images_list:
        cv2.imshow('Generated Film', img)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        # pass
    last_image = img
    images_list = []  # pulisco il vettore transizione