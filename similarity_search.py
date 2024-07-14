from sklearn.neighbors import NearestNeighbors
import numpy as np
import os
from feature_extraction import extract_features
def build_similarity_model(products, feature_extractor, image_folder):
    nn_models = {}
    image_paths = {}

    for product in products:
        try:
            product_id = str(product['_id'])
            product_title = product['title']

            print(f"Processing product: {product_id}, {product_title}")


            for category_name in os.listdir(image_folder):
                category_folder = os.path.join(image_folder, category_name)

                if not os.path.isdir(category_folder):
                    continue

                features = []
                paths = []

                for img_file in os.listdir(category_folder):

                    if img_file.startswith(product_id):
                        img_path = os.path.join(category_folder, img_file)
                        features.append(extract_features(img_path, feature_extractor))
                        paths.append(img_path)

                if features:
                    features = np.array(features).squeeze()
                    if features.ndim == 1:
                        features = features.reshape(1, -1)


                    n_neighbors = min(10, len(features))
                    nn_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(features)
                    nn_models[category_name] = nn_model
                    image_paths[category_name] = paths

        except KeyError as ke:
            print(f"KeyError processing product {product.get('_id', 'Unknown ID')}: {ke}")
        except Exception as e:
            print(f"Error processing product {product.get('_id', 'Unknown ID')}: {e}")

    return nn_models, image_paths

def find_similar_images(image_np, feature_extractor, model, nn_models, image_paths, class_indices):
    try:

        features = feature_extractor.predict(image_np)


        features = features.reshape(features.shape[0], -1)
        preds = model.predict(image_np)
        top_3_indices = np.argsort(preds[0])[-3:][::-1]
        top_3_categories = [list(class_indices.keys())[i] for i in top_3_indices]

        similar_images = []
        for category in top_3_categories:
            if category in nn_models:
                n_neighbors = nn_models[category].n_neighbors
                distances, indices = nn_models[category].kneighbors(features, n_neighbors=n_neighbors)
                for i in indices[0]:
                    similar_images.append(image_paths[category][i])
        unique_product_ids = []
        seen_product_ids = set()
        index = 0
        while len(unique_product_ids) < 5:
            if index >= len(similar_images):
                break
            img_path = similar_images[index]
            product_id = os.path.basename(img_path).split('_')[0]
            if product_id not in seen_product_ids:
                unique_product_ids.append(product_id)
                seen_product_ids.add(product_id)
            index += 1
        while len(unique_product_ids) < 5 and similar_images:
            unique_product_ids.append(unique_product_ids[len(unique_product_ids) % len(seen_product_ids)])
        return top_3_categories, unique_product_ids
    except Exception as e:
        print(f"Error in find_similar_images: {e}")
        return [], []