package uz.d2c.screens

import app.cash.paparazzi.DeviceConfig
import app.cash.paparazzi.Paparazzi
import app.cash.paparazzi.Snapshot
import app.cash.paparazzi.SnapshotHandler
import com.android.resources.Density
import org.junit.Rule
import org.junit.Test
import uz.d2c.screens.gen.ALL_SCREENS
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

/** Har bir snapshot'ni asl o'lchamda (1080x1704) PNG qilib saqlaydi. */
class FullResHandler(private val dir: File) : SnapshotHandler {
    override fun newFrameHandler(snapshot: Snapshot, frameCount: Int, fps: Int) = object : SnapshotHandler.FrameHandler {
        override fun handle(image: BufferedImage) {
            ImageIO.write(image, "png", File(dir, "${snapshot.name}.png"))
        }
        override fun close() {}
    }
    override fun close() {}
}

class RenderAllTest {
    private val out = File(System.getProperty("d2c.out")).apply { mkdirs() }

    @get:Rule
    val paparazzi = Paparazzi(
        deviceConfig = DeviceConfig(screenWidth = 1080, screenHeight = 1704, xdpi = 420, ydpi = 420, density = Density.create(420)),
        showSystemUi = false,
        snapshotHandler = FullResHandler(out),
    )

    @Test
    fun renderAll() {
        val log = File(out, "failures.txt")
        for ((id, content) in ALL_SCREENS) {
            try {
                paparazzi.snapshot(name = id.toString()) { content() }
            } catch (t: Throwable) {
                log.appendText("$id\t${t.javaClass.simpleName}: ${t.message?.lineSequence()?.firstOrNull()}\n")
            }
        }
    }
}
