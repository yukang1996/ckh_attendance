from deepface import DeepFace

dfs = DeepFace.find(img_path = "/Users/yukangwong/ckh_project/data/employees_db/yukang/yukang_1757528506.jpg", db_path = "/Users/yukangwong/ckh_project/data/employees_db")
print(dfs)
