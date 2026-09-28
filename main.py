import sys
from real_estate_regression import run_real_estate_pipeline, interactive_prediction
from mnist_cnn_classification import run_mnist_cnn, interactive_cnn_prediction


def main():
    while True:
        print("=====================================================")
        print("  Machine Learning & Image Classification Engines    ")
        print("=====================================================")
        print("Select an engine to initialize:")
        print("1. Train: Real Estate Regression Pipeline")
        print("2. Train: MNIST CNN Optimization Engine")
        print("3. Predict: Real Estate House Value (Interactive)")
        print("4. Predict: CNN Digit Recognition (Random Sample)")
        print("5. Exit")
        print("=====================================================")

        choice = input("Enter your choice (1-5): ")

        if choice == '1':
            print("\nInitializing Regression Pipeline...\n")
            run_real_estate_pipeline()
        elif choice == '2':
            print("\nInitializing CNN Optimization Engine...\n")
            run_mnist_cnn()
        elif choice == '3':
            interactive_prediction()
        elif choice == '4':
            interactive_cnn_prediction()
        elif choice == '5':
            print("Exiting system.")
            sys.exit(0)
        else:
            print("\nInvalid selection. Please try again.\n")


if __name__ == "__main__":
    main()