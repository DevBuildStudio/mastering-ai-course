import { cp, mkdir, readdir, readFile, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const portalRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const repositoryRoot = path.resolve(portalRoot, '..')
const pagesRoot = path.join(portalRoot, 'pages')

const courses = [
  {
    source: 'course1',
    route: 'course1',
    title: 'Course 1 · Foundations of AI Engineering',
  },
  {
    source: 'course2',
    route: 'course2',
    title: 'Course 2 · Building AI Systems',
  },
  {
    source: 'course3',
    route: 'course3',
    title: 'Course 3 · Production AI Engineering',
  },
]

function titleFromMarkdown(markdown, fallback) {
  const heading = markdown.match(/^#\s+(.+)$/m)?.[1]
  return heading?.replace(/\*\*/g, '').trim() || fallback
}

// Source READMEs use GitHub-relative links (e.g. `week1-foo.md`) so they browse
// correctly on GitHub. Rewrite them to absolute portal routes so they resolve
// correctly regardless of which page they're rendered on.
function rewriteRelativeLinks(markdown, route) {
  return markdown.replace(
    /\]\((week\d+-[a-z0-9-]+)\.md\)/gi,
    (_match, slug) => `](/${route}/${slug})`,
  )
}

function metadataModule(entries) {
  return `export default ${JSON.stringify(entries, null, 2)}\n`
}

async function resetDirectory(directory) {
  await rm(directory, { recursive: true, force: true })
  await mkdir(directory, { recursive: true })
}

async function syncCourse(course) {
  const sourceDirectory = path.join(repositoryRoot, course.source)
  const destinationDirectory = path.join(pagesRoot, course.route)
  await resetDirectory(destinationDirectory)

  const filenames = (await readdir(sourceDirectory))
    .filter((filename) => /^week\d+-.+\.md$/i.test(filename) && !/\scopy\.md$/i.test(filename))
    .sort((left, right) => left.localeCompare(right, undefined, { numeric: true }))

  const overviewSource = path.join(sourceDirectory, 'README.md')
  const overviewMarkdown = await readFile(overviewSource, 'utf8')
  await writeFile(
    path.join(destinationDirectory, 'index.md'),
    rewriteRelativeLinks(overviewMarkdown, course.route),
    'utf8',
  )

  const metadata = { index: 'Course Overview' }
  for (const filename of filenames) {
    const source = path.join(sourceDirectory, filename)
    const destination = path.join(destinationDirectory, filename)
    const markdown = await readFile(source, 'utf8')
    await writeFile(destination, rewriteRelativeLinks(markdown, course.route), 'utf8')
    metadata[path.basename(filename, '.md')] = titleFromMarkdown(markdown, filename)
  }

  await writeFile(
    path.join(destinationDirectory, '_meta.ts'),
    metadataModule(metadata),
    'utf8',
  )

  return { title: course.title, lessons: filenames.length }
}

async function syncWorkshop() {
  const sourceDirectory = path.join(repositoryRoot, 'workshops', 'fde-agent-workshop')
  const workshopsDirectory = path.join(pagesRoot, 'workshops')
  const destinationDirectory = path.join(workshopsDirectory, 'fde-agent-workshop')
  await resetDirectory(workshopsDirectory)
  await mkdir(destinationDirectory, { recursive: true })

  const filenames = (await readdir(sourceDirectory))
    .filter((filename) => /^chapter\d+-.+\.md$/i.test(filename))
    .sort((left, right) => left.localeCompare(right, undefined, { numeric: true }))

  await cp(
    path.join(sourceDirectory, 'workshop_training_material.md'),
    path.join(destinationDirectory, 'index.md'),
  )

  const metadata = { index: 'Workshop Overview' }
  for (const filename of filenames) {
    const source = path.join(sourceDirectory, filename)
    const markdown = await readFile(source, 'utf8')
    await cp(source, path.join(destinationDirectory, filename))
    metadata[path.basename(filename, '.md')] = titleFromMarkdown(markdown, filename)
  }

  await writeFile(
    path.join(destinationDirectory, '_meta.ts'),
    metadataModule(metadata),
    'utf8',
  )
  await writeFile(
    path.join(workshopsDirectory, '_meta.ts'),
    metadataModule({ 'fde-agent-workshop': 'Forward Deployed AI Engineering' }),
    'utf8',
  )

  return { title: 'Forward Deployed AI Engineering Workshop', lessons: filenames.length }
}

const results = []
for (const course of courses) {
  results.push(await syncCourse(course))
}
results.push(await syncWorkshop())

for (const result of results) {
  console.log(`Synced ${result.lessons} lessons: ${result.title}`)
}