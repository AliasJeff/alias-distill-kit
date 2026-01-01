import sys
from distillkit.main import train_teacher_main, main, evaluate_main, infer_main

if __name__ == "__main__":
    command = sys.argv[1]
    sys.argv.pop(1)

    print(f"DEBUG: Running command '{command}' with args: {sys.argv[1:]}")

    if command == "train-teacher":
        train_teacher_main()
    elif command == "distill":
        main()
    elif command == "evaluate":
        evaluate_main()
    elif command == "infer":
        infer_main()
    else:
        print(f"Unknown command: {command}")
