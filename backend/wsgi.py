import sys
import traceback

try:
    from app import create_app
    app = create_app()
except Exception as e:
    print(f"ERROR: Failed to create app: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    app.run()

