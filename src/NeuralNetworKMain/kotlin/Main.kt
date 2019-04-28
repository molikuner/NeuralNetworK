package com.molikuner.neuralnetwork

import kotlin.system.measureTimeMicros

fun main() {
    val network = NetworK {
        inputLayer()
        hiddenLayer()
        outputLayer()
    }
    println(network.toLogString())
    println(network.run(0.0, 1.0))

    // Testing
    measure("\nNetworK create took: @-@micros") {
        NetworK {
            inputLayer {
                numInputs = 5
                layerBias = BiasNeuron()
            }
            hiddenLayers(20)
            outputLayer {
                numOutputs = 4
            }
        }
    }
}

inline fun <T : Any> measure(message: String, block: () -> T): T {
    lateinit var x: T
    println(message.replace("@-@", measureTimeMicros {
        x = block()
    }.toString()))
    return x
}
