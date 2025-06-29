#!/bin/bash

# EchoWear Docker Management Script
# This script provides convenient commands to run different EchoWear services

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  EchoWear Docker Management${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Function to build the Docker image
build_image() {
    print_status "Building EchoWear Docker image..."
    docker-compose build
    print_status "Docker image built successfully!"
}

# Function to run the GUI
run_gui() {
    print_status "Starting EchoWear GUI..."
    print_warning "Make sure X11 forwarding is enabled for GUI support"
    docker-compose up echowear-gui
}

# Function to run tests
run_tests() {
    print_status "Running EchoWear tests..."
    docker-compose up echowear-test
}

# Function to run camera test
run_camera_test() {
    print_status "Running camera test..."
    docker-compose up echowear-camera-test
}

# Function to run demo
run_demo() {
    print_status "Running EchoWear demo..."
    docker-compose up echowear-demo
}

# Function to run scene graph server
run_scene_graph_server() {
    print_status "Starting scene graph server..."
    docker-compose up scene-graph-server
}

# Function to run all services
run_all() {
    print_status "Starting all EchoWear services..."
    docker-compose up
}

# Function to stop all services
stop_all() {
    print_status "Stopping all EchoWear services..."
    docker-compose down
}

# Function to clean up
cleanup() {
    print_status "Cleaning up Docker resources..."
    docker-compose down --volumes --remove-orphans
    docker system prune -f
    print_status "Cleanup completed!"
}

# Function to show logs
show_logs() {
    local service=${1:-echowear-gui}
    print_status "Showing logs for $service..."
    docker-compose logs -f $service
}

# Function to show status
show_status() {
    print_status "Docker containers status:"
    docker-compose ps
}

# Function to show help
show_help() {
    print_header
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build           Build the Docker image"
    echo "  gui             Run the EchoWear GUI"
    echo "  test            Run tests"
    echo "  camera-test     Run camera test"
    echo "  demo            Run the enhanced demo"
    echo "  server          Run scene graph server"
    echo "  all             Run all services"
    echo "  stop            Stop all services"
    echo "  cleanup         Clean up Docker resources"
    echo "  logs [SERVICE]  Show logs (default: echowear-gui)"
    echo "  status          Show container status"
    echo "  help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build        # Build the image"
    echo "  $0 gui          # Run the GUI"
    echo "  $0 logs         # Show GUI logs"
    echo "  $0 logs server  # Show server logs"
}

# Main script logic
main() {
    check_docker
    
    case "${1:-help}" in
        build)
            build_image
            ;;
        gui)
            run_gui
            ;;
        test)
            run_tests
            ;;
        camera-test)
            run_camera_test
            ;;
        demo)
            run_demo
            ;;
        server)
            run_scene_graph_server
            ;;
        all)
            run_all
            ;;
        stop)
            stop_all
            ;;
        cleanup)
            cleanup
            ;;
        logs)
            show_logs "$2"
            ;;
        status)
            show_status
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@" 