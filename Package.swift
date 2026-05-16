// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "mlx-swift-diffrast",
    platforms: [
        .macOS(.v14),
        .iOS(.v16),
        .visionOS(.v1),
    ],
    products: [
        .library(name: "MLXDiffRast", targets: ["MLXDiffRast"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", exact: "0.31.3"),
    ],
    targets: [
        .target(
            name: "MLXDiffRast",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "MLXDiffRastTests",
            dependencies: [
                "MLXDiffRast",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ]
        ),
    ]
)
