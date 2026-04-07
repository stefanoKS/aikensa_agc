CREATE DATABASE IF NOT EXISTS aikensa_agc
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

USE aikensa_agc;

CREATE TABLE IF NOT EXISTS inspection_results (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    partName INT NOT NULL,
    lotNumber VARCHAR(255) NOT NULL,
    serialNumber VARCHAR(255) NOT NULL,
    ok_add INT NOT NULL DEFAULT 0,
    ng_add INT NOT NULL DEFAULT 0,
    timestamp DATETIME NOT NULL,
    kensainName VARCHAR(255),
    UNIQUE KEY uq_part_lot_serial (partName, lotNumber, serialNumber),
    KEY idx_part_lot_time (partName, lotNumber, timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Optional if the application user already exists on the server:
-- GRANT ALL PRIVILEGES ON aikensa_agc.* TO 'AIKENSAAGC'@'%';
-- FLUSH PRIVILEGES;