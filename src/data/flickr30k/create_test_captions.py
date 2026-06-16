def create_test_captions(test_df):
    test_captions={}

    for index, row in test_df.iterrows():
        img_id=row['image_name']
        cap=row[' comment']
        
        test_captions.setdefault(img_id, []).append(cap.lower())
    return test_captions