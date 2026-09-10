// Drop-in replacement for City3D's code/CLI_Example_1/main.cpp.
#include "../model/map.h"
#include "../model/map_io.h"
#include "../model/point_set.h"
#include "../model/point_set_io.h"
#include "../method/method_global.h"
#include "../method/reconstruction.h"

#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: urbanstock-city3d POINT_CLOUD FOOTPRINT OUTPUT_OBJ\n";
        return EXIT_FAILURE;
    }

    Method::min_points = 40;
    Method::pixel_size = 0.15;
    const std::string input_cloud_file = argv[1];
    const std::string input_footprint_file = argv[2];
    const std::string output_file = argv[3];

    PointSet* point_set = PointSetIO::read(input_cloud_file);
    if (!point_set) {
        std::cerr << "failed to load point cloud: " << input_cloud_file << '\n';
        return EXIT_FAILURE;
    }
    const vec3& offset = point_set->offset();
    Map* footprint = MapIO::read(
        input_footprint_file,
        vec3(offset.x, offset.y, -point_set->bbox().z_min())
    );
    if (!footprint) {
        std::cerr << "failed to load footprint: " << input_footprint_file << '\n';
        delete point_set;
        return EXIT_FAILURE;
    }

    Reconstruction reconstruction;
    reconstruction.segmentation(point_set, footprint);
    if (!reconstruction.extract_roofs(point_set, footprint)) {
        std::cerr << "no roof planes could be extracted\n";
        delete point_set;
        delete footprint;
        return EXIT_FAILURE;
    }

    Map* result = new Map;
#ifdef HAS_GUROBI
    const bool reconstructed = reconstruction.reconstruct(
        point_set, footprint, result, LinearProgramSolver::GUROBI
    );
#else
    const bool reconstructed = reconstruction.reconstruct(
        point_set, footprint, result, LinearProgramSolver::SCIP
    );
#endif
    const bool saved = reconstructed && result->size_of_facets() > 0
        && MapIO::save(output_file, result);

    delete point_set;
    delete footprint;
    delete result;
    if (!saved) {
        std::cerr << "City3D reconstruction failed\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
