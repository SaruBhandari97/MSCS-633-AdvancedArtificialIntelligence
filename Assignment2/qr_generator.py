# =============================================================
# QR Code Generator
# Author: Saru Bhandari
# Date: 2026
# Description: Generates a QR code image from a user-provided
#              URL using the qrcode and Pillow libraries.
# =============================================================

import qrcode
from PIL import Image

def generate_qr_code(url: str, filename: str = "qr_output.png") -> None:
    """
    Generates a QR code for the given URL and saves it as an image file.

    Args:
        url (str): The URL to encode into the QR code.
        filename (str): The output filename for the QR code image.
    """

    # Validate that the URL is not empty
    if not url.strip():
        print("Error: URL cannot be empty.")
        return

    # Configure QR code settings
    qr = qrcode.QRCode(
        version=1,                          # Controls size (1 = smallest, 40 = largest)
        error_correction=qrcode.constants.ERROR_CORRECT_H,  # High error correction
        box_size=10,                        # Size of each box in pixels
        border=4,                           # Thickness of the border (in boxes)
    )

    # Add the URL data to the QR code
    qr.add_data(url)
    qr.make(fit=True)  # Automatically adjust version to fit data

    # Generate the QR code image with black and white colors
    img = qr.make_image(fill_color="black", back_color="white")

    # Save the image to the specified filename
    img.save(filename)
    print(f"QR code successfully generated: '{filename}'")
    print(f"Encoded URL: {url}")

    # Display the image
    img.show()


def main():
    """
    Main function to run the QR Code Generator application.
    Prompts the user to input a URL and generates a QR code.
    """

    print("=" * 40)
    print("       QR Code Generator App")
    print("  Powered by: qrcode + Pillow")
    print("=" * 40)

    # Prompt user for URL input
    url = input("\nEnter the URL to generate QR code for:\n> ").strip()

    # Generate the QR code
    generate_qr_code(url, filename="qr_output.png")

    print("\nDone! Check your project folder for 'qr_output.png'.")


# Entry point of the program
if __name__ == "__main__":
    main()