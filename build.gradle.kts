plugins {
    kotlin("multiplatform") version "1.3.31"
}

repositories {
    mavenCentral()
}

kotlin {
    // For ARM, preset function should be changed to iosArm32() or iosArm64()
    // For Linux, preset function should be changed to e.g. linuxX64()
    // For MacOS, preset function should be changed to e.g. macosX64()
    linuxX64("NeuralNetworK") {
        binaries {
            // Comment the next section to generate Kotlin/Native library (KLIB) instead of executable file:
            executable("NeuralNetworKApp") {
                // Change to specify fully qualified name of your application's entry point:
                entryPoint = "com.molikuner.neuralnetwork.main"
            }
        }
    }
}

// TODO implement coroutines

// Use the following Gradle tasks to run your application:
// :runNeuralNetworKAppReleaseExecutableNeuralNetworK - without debug symbols
// :runNeuralNetworKAppDebugExecutableNeuralNetworK - with debug symbols
