set(INCLUDE_DIRECTORIES
${ROOT_INCLUDE_DIR}  
${CUDA_INCLUDE}
${CMAKE_SOURCE_DIR}/base 
${CMAKE_SOURCE_DIR}/mpdfield 
${CMAKE_SOURCE_DIR}/fairtools
${CMAKE_SOURCE_DIR}/kalman
${CMAKE_SOURCE_DIR}/mpdbase
)

set(CUDA_INCLUDE_DIRECTORIES 
#${CUDA_INCLUDE}
"${CMAKE_CURRENT_SOURCE_DIR}"
)



include_directories( ${INCLUDE_DIRECTORIES} ${CUDA_INCLUDE_DIRECTORIES})

set(LINK_DIRECTORIES
${ROOT_LIBRARY_DIR}
${CUDA_LIBRARIES}

)




set(MnvertLocal_SRCS MnvertLocal.cu )

#set(MnvertLocal_HEADERS MnvertLocal.h )

#set(MnvertLocal_LINKDEF  MnvertLocalLinkDef.h)
#set(MnvertLocal_DICTIONARY ${CMAKE_CURRENT_BINARY_DIR}/MyMnvertLocalDict.cxx) 

#ROOT_GENERATE_DICTIONARY("${MnvertLocal_HEADERS}" "${MnvertLocal_LINKDEF}" "${MnvertLocal_DICTIONARY}" "${INCLUDE_DIRECTORIES}")

cuda_add_library(MnvertLocal SHARED ${MnvertLocal_SRCS})

CUDA_COMPILE(CUDA_FILES )
#target_link_libraries(cudaMnvertLocal ${CUDA_LIBRARIES} ${ROOT_LIBRARIES})
#CUDA_BUILD_CLEAN_TARGET()
###########################################

###########################################

#set(MNVERTLOCALWRAPPER_SRCS 
#MnvertLocalWrapper.cxx
#)


#set(MNVERTLOCALWRAPPER_HEADERS
#MnvertLocalWrapper.h
#)

#set(MnvertLocal_LINKDEF  MnvertLocalLinkDef.h)
#set(MnvertLocal_DICTIONARY ${CMAKE_CURRENT_BINARY_DIR}/MyMnvertLocalWrapperDict.cxx) 

#ROOT_GENERATE_DICTIONARY("${MNVERTLOCALWRAPPER_HEADERS}" "${MnvertLocal_LINKDEF}" "${MnvertLocal_DICTIONARY}" "${INCLUDE_DIRECTORIES}")

#set(MNVERTLOCALWRAPPER_SRCS 
#${MNVERTLOCALWRAPPER_SRCS}
#${MnvertLocal_DICTIONARY}
#)

#add_library(MnvertLocalWrapper SHARED ${MNVERTLOCALWRAPPER_SRCS})
#add_dependencies(MnvertLocalWrapper MnvertLocal)
#target_link_libraries(MnvertLocalWrapper   ${CMAKE_BINARY_DIR}/lib/libMnvertLocal.so "${CUDA_TARGET_LINK}"
#   "${CUDA_CUT_TARGET_LINK}"  ${ROOT_LIBRARIES})

#set_target_properties (MnvertLocalWrapper PROPERTIES ${ROOT_LIBRARIES_PROPERTIES})
################ install ###################
#install(TARGETS MnvertLocalWrapper DESTINATION ${CMAKE_BINARY_DIR}/lib)
