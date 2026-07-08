idf_build_get_property(component_targets __COMPONENT_TARGETS)
if ("___idf_espressif__esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
elseif("___idf_esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir esp-dl COMPONENT_DIR)
endif()
set(cmake_dir ${espdl_dir}/fbs_loader/cmake)
include(${espdl_dir}/fbs_loader/cmake/utilities.cmake)

idf_component_register(
    SRCS "human_pose_detect.cpp"
    INCLUDE_DIRS "."
    REQUIRES esp-dl
)

target_add_aligned_binary_data(
    ${COMPONENT_LIB}
    ${COMPONENT_DIR}/models/p4/movenet_lightning_192_int8.espdl
    BINARY
    DEPENDS ${COMPONENT_DIR}/models/p4/movenet_lightning_192_int8.espdl
)
