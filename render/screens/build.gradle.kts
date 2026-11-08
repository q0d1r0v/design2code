plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
    id("app.cash.paparazzi")
}
android {
    namespace = "uz.d2c.screens"
    compileSdk = 34
    defaultConfig { minSdk = 24 }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    buildFeatures { compose = true }
    testOptions { unitTests.all { it.maxHeapSize = "3g" } }
}
kotlin {
    jvmToolchain(17)
    // imtihon: Material3/Foundation "experimental" API'lari uchun @OptIn talab qilinmasin (barcha modellarga bir xil)
    compilerOptions {
        optIn.addAll("androidx.compose.material3.ExperimentalMaterial3Api",
                     "androidx.compose.foundation.ExperimentalFoundationApi",
                     "androidx.compose.foundation.layout.ExperimentalLayoutApi")
    }
}
dependencies {
    implementation(platform("androidx.compose:compose-bom:2024.09.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material3:material3")
    testImplementation("junit:junit:4.13.2")
    testCompileOnly(files("../../tools/awtstub/awt-compile-stub.jar"))
}

tasks.withType<Test>().configureEach {
    systemProperty("d2c.out", (project.findProperty("d2cOut") ?: "").toString())
    outputs.upToDateWhen { false }
}
