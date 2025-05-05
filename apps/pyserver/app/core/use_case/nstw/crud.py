from PIL import Image
import io
import opennsfw2 as n2


class CRUDNstw:
    @staticmethod
    def detect_nsfw(image_bytes: bytes):
        # Carrega a imagem a partir dos bytes
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocessa a imagem conforme o modelo
        processed_image = n2.preprocess_image(image, n2.Preprocessing.YAHOO)

        # Cria o modelo
        model = n2.make_open_nsfw_model()

        # Realiza a predição
        prediction = model.predict(processed_image[None, ...])[0][0]

        # Define o limiar para considerar como NSFW
        threshold = 0.7
        is_nsfw = bool(prediction >= threshold)
        detail = f"Probabilidade NSFW: {prediction:.2f}"

        return {"nsfw": is_nsfw, "detail": detail}
