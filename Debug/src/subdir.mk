################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/Face\ Recognition.cpp \
../src/histogram.cpp \
../src/lbp.cpp 

OBJS += \
./src/Face\ Recognition.o \
./src/histogram.o \
./src/lbp.o 

CPP_DEPS += \
./src/Face\ Recognition.d \
./src/histogram.d \
./src/lbp.d 


# Each subdirectory must supply rules for building sources it contributes
src/Face\ Recognition.o: ../src/Face\ Recognition.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/usr/local/include/opencv -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"src/Face Recognition.d" -MT"src/Face\ Recognition.d" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/usr/local/include/opencv -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


