from deepface import DeepFace

def compare_faces(id_img_path, selfie_img_path):
    try:
        result = DeepFace.verify(
            img1_path=id_img_path,
            img2_path=selfie_img_path,
            enforce_detection=False
        )

        similarity = 1 - result["distance"]

        return {
            "match": result["verified"],
            "similarity": round(similarity, 3),
            "model": "DeepFace"
        }

    except Exception as e:
        return {
            "match": False,
            "error": str(e)
        }


if __name__ == "__main__":
    res = compare_faces(
        "sample_data/mock_aadhaar_genuine.jpg",
        "sample_data/selfie_match.jpg"
    )
    print(res)