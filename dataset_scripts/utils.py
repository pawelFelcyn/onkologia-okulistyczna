import os

def get_all_labeled_images() -> list[tuple[str, str, str]]:
    ret = []
    root = os.path.join('Ophthalmic_Scans', 'raw')
    
    for patient_dir in [x for x in os.listdir(root) if x.startswith('sub-')]:
        patient_path = os.path.join(root, patient_dir)
        if not os.path.isdir(patient_path):
            continue

        for session_dir in [x for x in os.listdir(patient_path) if x.startswith('ses-')]:
            session_path = os.path.join(patient_path, session_dir)
            if not os.path.isdir(session_path):
                continue

            for kind in os.listdir(session_path):
                kind_path = os.path.join(session_path, kind)
                if not os.path.isdir(kind_path):
                    continue

                for area_dir in os.listdir(kind_path):
                    area_path = os.path.join(kind_path, area_dir)
                    if not os.path.isdir(area_path):
                        continue

                    for eye_dir in [x for x in os.listdir(area_path) if x in ['R', 'L']]:
                        labels_path = os.path.join(area_path, eye_dir, 'labels')
                        if not os.path.isdir(labels_path):
                            continue

                        for filename in [x for x in os.listdir(labels_path) if x.endswith('.txt')]:
                            label_file = os.path.join(labels_path, filename)
                            tumor_mask = os.path.join(labels_path.replace('labels', 'masks'), 'tumor', filename.replace('.txt', '.png'))
                            fluid_mask = os.path.join(labels_path.replace('labels', 'masks'), 'fluid', filename.replace('.txt', '.png'))

                            resized_images_dir = os.path.join(labels_path.replace('labels', 'resized_images'))
                            res_png = os.path.join(resized_images_dir, filename.replace('.txt', '.png'))
                            res_jpg = os.path.join(resized_images_dir, filename.replace('.txt', '.jpg'))
                            res = res_png if os.path.isfile(res_png) else res_jpg if os.path.isfile(res_jpg) else None

                            original_images_dir = os.path.join(labels_path.replace('labels', 'original_images'))
                            orig_png = os.path.join(original_images_dir, filename.replace('.txt', '.png'))
                            orig_jpg = os.path.join(original_images_dir, filename.replace('.txt', '.jpg'))
                            orig = orig_png if os.path.isfile(orig_png) else orig_jpg if os.path.isfile(orig_jpg) else None

                            if orig and res and os.path.isfile(tumor_mask) and os.path.isfile(fluid_mask):
                                ret.append((orig, res, label_file, tumor_mask, fluid_mask))
    
    return ret

def get_all_original_images():
    root = os.path.join('Ophthalmic_Scans', 'raw')
    results = []

    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) == "original_images":
            for fname in filenames:
                if fname.lower().endswith(('.jpg', '.png')):
                    full_path = os.path.join(dirpath, fname)
                    results.append(full_path)
    return results

def get_all_resized_images(root: str = 'raw'):
    root = os.path.join('Ophthalmic_Scans', root)
    results = []

    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) == "resized_images":
            for fname in filenames:
                if fname.lower().endswith(('.jpg', '.png')):
                    full_path = os.path.join(dirpath, fname)
                    results.append(full_path)
    return results

if __name__ == '__main__':
    c = get_all_labeled_images()
    print("Found {} labeled images.".format(len(c)))