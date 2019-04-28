package com.molikuner.neuralnetwork

import kotlin.math.E
import kotlin.math.pow
import kotlin.random.Random

typealias Weight = Double
typealias Activation = Double
val Activation.sigmoid: Activation
    get() = 1 / (1 + E.pow(-this))

sealed class Neuron {
    override fun toString(): String = "${this::class.simpleName}"
}
sealed class DefinedNeuron : Neuron()
sealed class PredefinedNeuron(internal open val activation: Activation) : DefinedNeuron() {
    val asActivatedNeuron: ActivatedNeuron
        get() = ActivatedNeuron(activation, this)

    override fun toString(): String = "${this::class.simpleName} { $activation }"
}

class InputNeuron(internal var _input: Activation) : PredefinedNeuron(_input) {
    override val activation: Activation
        get() = input
    val input: Activation
        get() = _input
}

class WeightableNeuron : DefinedNeuron() {
    private val weights: MutableMap<DefinedNeuron, Weight> = mutableMapOf()
    fun weightTo(item: ActivatedNeuron): Weight = weights[item.origNeuron]
            ?: Random.nextDouble(-10.0, 10.0).also { weights[item.origNeuron] = it }

    override fun toString(): String = "${this::class.simpleName} { ${weights.values.joinToString(" / ")} }"
}
typealias HiddenNeuron = WeightableNeuron
typealias OutputNeuron = WeightableNeuron

class BiasNeuron private constructor(val bias: Activation) : PredefinedNeuron(bias) {
    companion object {
        inline operator fun invoke(block: BiasNeuronBuilder.() -> Unit = {}): BiasNeuron = BiasNeuronBuilder().apply(block).build()
    }

    class BiasNeuronBuilder {
        var bias: Activation = 1.0

        fun build(): BiasNeuron = BiasNeuron(bias)
    }
}

class ActivatedNeuron(val activation: Activation, val origNeuron: DefinedNeuron) : Neuron() {
    override fun toString(): String = "${this::class.simpleName} { $origNeuron -> $activation }"
}
