import requests
import tensorflow as tf

class Mlapi:
    def __init__(self):
        self.class_names_gender = ['man', 'woman']
        self.model_path = ".\Gender-Recognition Model"
    
    def load_and_prep(self, filepath):
        img = tf.io.read_file(filepath)
        img = tf.io.decode_image(img)
        img = tf.image.resize(img, (224, 224))

        return img
    
    def prep_url_img(self, url):
        img = requests.get(url, stream=True)
        img = tf.io.encode_jpeg(img)
        img = tf.io.decode_image(img)
        img = tf.image.resize(img, (224, 224))

        return img
    
    def load_and_pred_model(self, img_path):
        model = tf.keras.models.load_model(self.model_path)
        img = self.load_and_prep(img_path)
        img = tf.expand_dims(img, axis=0)
        pred = model.predict(img)[0]

        return pred
    
    def url_pred(self, url):
        model = tf.keras.models.load_model(self.model_path)
        img = self.prep_url_img(url)
        img = tf.expand_dims(img, axis=0)
        pred = model.predict(img)[0]

        return img.shape
    
    def gen_null_https_url(self, url):
        url_lit = url.split("//")[1].replace("/", "-&-").replace(".", "---")

        return url_lit
    
    def gen_https_url(self, url):
        url_lit = url.replace("-&-", "/").replace("---", ".")
        url_lit = f"https://{url_lit}"

        return url_lit

    def img_gen(self, url):
        https_url = self.gen_https_url(url)

        response = requests.get(https_url)
        if response.status_code == 200:
            with open("./Responses/sample.jpg", 'wb') as f:
                f.write(response.content)
    
    def make_pred(self, url):
        self.img_gen(url)
        pred_prob_gender = self.load_and_pred_model("./Responses/sample.jpg")

        pred_class_gender = self.class_names_gender[pred_prob_gender.argmax()]

        pred = {
            "pred-class": pred_class_gender,
            "pred-prob-man": str(pred_prob_gender[0]),
            "pred-prob-woman": str(pred_prob_gender[1]),
            "img-url": self.gen_https_url(url)
        }

        return pred



if __name__ == "__main__":
    api = Mlapi()
    # url = "https://pngimg.com/uploads/girls/small/girls_PNG6492.png"
    url = "https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=687&q=80"
    # url = "https://raw.githubusercontent.com/Hrushi11/Gender-Recognition/main/images/lady.jpg"
    null_https_url = api.gen_null_https_url(url)
    print(null_https_url)
    # print(api.make_pred(null_https_url))

    # pred = api.make_pred(null_https_url)
    # print(null_https_url)
    # api.img_gen(null_https_url)