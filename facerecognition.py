import face_recognition

# 1. Load the images
known_image = face_recognition.load_image_file("sampel.jpg")
unknown_image = face_recognition.load_image_file("sampel2.jpg")

# 2. Extract 128-dimensional face encodings
# [0] fetches the first face found in the image
known_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

# 3. Compare the faces
# The function expects a list of known encodings as the first parameter
results = face_recognition.compare_faces([known_encoding], unknown_encoding, tolerance=0.6)

# 4. Print results
if results[0]:
    print("Match found: It is the same person!")
else:
    print("No match: These are different people.")
    
# 5. Optional: Get the exact Euclidean distance
face_distance = face_recognition.face_distance([known_encoding], unknown_encoding)
print(f"Distance between faces: {face_distance[0]:.2f}")
