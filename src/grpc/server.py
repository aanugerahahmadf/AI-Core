"""
AI Core gRPC Server Entry Point
Wedding Flower Decoration App - CBIR gRPC Server

This is the main entry point for the AI Core gRPC server.
It starts the gRPC server on the configured port (default: 50052).

Usage:
    python -m grpc.server
    python -m grpc.server --port 50052 --workers 10
"""

import os
import sys
import argparse
import signal
import time
import hmac
from concurrent import futures

# Import grpcio BEFORE adding src/ to sys.path, otherwise the local
# `src/grpc` package shadows the installed grpcio module.
import grpc
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import generated protobuf classes
try:
    from .generated import cbir_pb2
    from .generated import cbir_pb2_grpc
except ImportError:
    try:
        from generated import cbir_pb2
        from generated import cbir_pb2_grpc
    except ImportError:
        print("Warning: Generated protobuf classes not found. Please run protoc first.")
        print("Run: python -m grpc_tools.protoc -I./protos --python_out=./generated --grpc_python_out=./generated ./protos/cbir.proto")
        sys.exit(1)

# Import service implementation
try:
    from .services.cbir_service import CbirServiceServicer
except ImportError:
    from services.cbir_service import CbirServiceServicer


class ApiKeyInterceptor(grpc.ServerInterceptor):
    def intercept_service(self, continuation, handler_call_details):
        handler = continuation(handler_call_details)
        expected = os.environ.get('AI_CORE_API_KEY', '')
        method = handler_call_details.method.rsplit('/', 1)[-1]
        if not expected or method == 'GetHealth' or handler is None:
            return handler

        def authenticated(request, context):
            metadata = dict(context.invocation_metadata())
            supplied = metadata.get('x-ai-core-key', '')
            if not hmac.compare_digest(expected, supplied):
                context.abort(grpc.StatusCode.UNAUTHENTICATED, 'AI Core authentication required')
            return handler.unary_unary(request, context)

        return grpc.unary_unary_rpc_method_handler(
            authenticated,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer,
        )


def create_server(port=50052, max_workers=10):
    """Create and configure the gRPC server."""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        interceptors=[ApiKeyInterceptor()],
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),  # 50MB
        ]
    )
    
    # Register CBIR service
    cbir_pb2_grpc.add_CbirServiceServicer_to_server(
        CbirServiceServicer(), server
    )
    
    # Bind to port
    host = os.environ.get('AI_CORE_GRPC_HOST', '127.0.0.1')
    # TLS credentials are deployment-specific and must be provided by the
    # service mesh/reverse proxy. Refuse a public bind without explicit opt-in.
    if host not in {'127.0.0.1', 'localhost', '::1'} and os.environ.get('AI_CORE_GRPC_ALLOW_INSECURE_PUBLIC') != 'true':
        raise RuntimeError('Refusing public insecure gRPC bind; use TLS or explicit private-network opt-in.')
    server.add_insecure_port(f'{host}:{port}')
    
    return server


def serve(port=50052, max_workers=10):
    """Start the gRPC server and wait for termination."""
    server = create_server(port, max_workers)
    server.start()
    
    print(f"=" * 60)
    print(f"AI Core gRPC Server Started")
    print(f"=" * 60)
    print(f"Port: {port}")
    print(f"Workers: {max_workers}")
    print(f"Services: CbirService")
    print(f"Press Ctrl+C to stop")
    print(f"=" * 60)
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\nShutting down AI Core gRPC server...")
        server.stop(0)
        print("Server stopped.")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        while True:
            time.sleep(86400)  # Sleep for 24 hours
    except KeyboardInterrupt:
        server.stop(0)
        print("Server stopped.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='AI Core gRPC Server - Wedding Flower Decoration App'
    )
    parser.add_argument(
        '--port', type=int, default=50052,
        help='gRPC server port (default: 50052)'
    )
    parser.add_argument(
        '--workers', type=int, default=10,
        help='Number of worker threads (default: 10)'
    )
    
    args = parser.parse_args()
    
    serve(port=args.port, max_workers=args.workers)


if __name__ == '__main__':
    main()
