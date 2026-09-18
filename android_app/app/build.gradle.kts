plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "org.sih26.deadreckoning"
    compileSdk = 34

    defaultConfig {
        applicationId = "org.sih26.deadreckoning"
        // MIP Section 7: min SDK 26 (Android 8.0), which is where a persistent
        // foreground-service notification became mandatory rather than optional -
        // exactly the mechanism this app depends on to keep sampling with the
        // screen off during a test drive.
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "0.1-demo"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        compose = true
    }
    // fusion/*.kt is deliberately Android-free (see its own file docs) so it can be
    // unit-tested on a desktop JVM via tools/parity/ and tools/phone_replay/ without
    // an emulator. Nothing here needs to change that; Gradle just compiles it as
    // part of the same module.
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.6")
    implementation("androidx.activity:activity-compose:1.9.2")
    implementation(platform("androidx.compose:compose-bom:2024.09.02"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui-tooling-preview")
    // Bottom-nav icons (Navigation, List, Info) live outside the core icon set.
    implementation("androidx.compose.material:material-icons-extended")

    // EJML was considered and deliberately not used - see the module doc at the top
    // of fusion/LinAlg.kt for the reasoning and the parity fixture that stands in
    // for the safety net a trusted library would have provided.

    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
}
