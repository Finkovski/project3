
# Data folder

Put source CSV(s) into `data/raw/`.

**Expected columns (auto-mapped if names differ):**
- `review_text` (e.g., `reviews.text`, `reviewText`, `text`)
- `rating` (e.g., `reviews.rating`, `stars`)
- Optional: `product_id` (`asin`), `product_title` (`name`, `title`), `category` (`categories`)

Example minimal CSV:

review_text,rating,product_title,category
"Great battery life and sharp screen",5,"Kindle Paperwhite","Ebook readers"
"Keys are mushy and the RGB died",2,"Mechanical Keyboard X","Accessories"
"Average",3,"Generic Battery Pack","Batteries"
