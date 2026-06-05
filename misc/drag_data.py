import shutil
from pathlib import Path

def reorganize_dicom_files() -> None:
    # Define your source and target root directories
    source_root = Path("./all_data")
    target_root = Path("./bilateral_standing_AP")

    # 1. Find all bi.dcm files matching your specific pattern
    # This looks exactly for data/ID***/X-ray/bi.dcm
    dcm_files = list(source_root.glob("ID*/X-ray/Bilateral_standing_AP_view.dcm"))

    if not dcm_files:
        print("No files found. Check your source path!")
        return

    print(f"Found {len(dcm_files)} files. Starting copy process...\n")

    for dcm_path in dcm_files:
        # 2. Extract the patient ID (e.g., "ID001")
        # dcm_path.parent is "X-ray"
        # dcm_path.parent.parent is "ID001"
        patient_id = dcm_path.parent.parent.name 
        # 4. Define the final destination file path

        new_filename = f"{patient_id}.dcm"

        target_file = target_root / new_filename
        # 5. Copy the file
        # shutil.copy2 is used instead of shutil.copy because it preserves
        # file metadata like creation and modification times
        shutil.copy2(dcm_path, target_file)
        print(f"Copied to: {target_file}") 

    print("\nData reorganization complete!")

if __name__ == "__main__":
    reorganize_dicom_files()