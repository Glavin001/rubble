use metal::*;

pub struct MetalDevice {
    pub device: Device,
    pub queue: CommandQueue,
    pub chip: String,
}

impl MetalDevice {
    pub fn new() -> anyhow::Result<Self> {
        let device = Device::system_default().ok_or_else(|| anyhow::anyhow!("no Metal device"))?;
        let name = device.name().to_string();
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            chip: name,
        })
    }
}
